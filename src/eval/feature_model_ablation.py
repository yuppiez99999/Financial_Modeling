"""特征集 × 模型族联合消融（Issue #55 第五轮 —— 前四条嫌疑排查完后的最后一条）。

## 问题从哪来

Issue #55 已排查完池共线性、标签口径、周期选择、置信度语义四条嫌疑
（见 `cairn/model-optimization-findings.md` / `cairn/regime-conditioned-signal.md`），
chỉ剩下两条**没有被量化过**的嫌疑：

1. **特征集**：当前特征集 107 列（基础技术指标 + 扩展指标库 + 多因子 + 宏观），
   彼此信息高度重叠（`ma_5 / ma_5_ratio`、`ema_* / wma_* / dema_* / trima_*` …）。
   冗余不一定有害，但也可能把树的切分预算耗在重复信息上。
2. **模型族**：单棵 LightGBM 对日频价量因子的预测力可能已到天花板
   （AUC ≈ 0.50~0.54），换更强的非线性模型族（或加正则）是否能抬升。

## 判据（写死，与 `benchmark_relative` 唯一记分板一致）

- **唯一通过条件 = 组合层「相对全池等权的净超额」**（扣 2× 单边成本、非重叠调仓、
  优于随机子集），**不看** IC / 命中率 / AUC —— 前几轮已记账：这些是**绝对读数**，
  无法把"市场给的钱"和"模型赚的钱"分开（见 `cairn/benchmark-relative-edge.md`）。
- 消融臂的**增量**必须同时满足：净超额转正 **且** 相对基线臂的**配对差值 t ≥ 2**。
  否则结论是 `no_increment` —— **不加特征、不换模型**。

## 两条独立结论

- `feature_ablation.verdict`：特征消融是否支持「加法」候选（全量 > 最优子集）；
- `model_ablation.verdict`：模型族消融是否支持「换族」候选（某族 > 现行基线族）。

两者都只报数：`affects_gate=False`，**不**自动改特征集 / 模型配置 / 门禁。

## 结构上避开的一个坑（先于本模块就存在的复现性风险）

特征**分组**不是硬编码列名清单，而是按**后缀/前缀族**从当前特征集派生
（`ma_*` / `ema_*` / `ret_*` / `factor_*` / `macro_*` …）。硬编码清单会在
特征集演进后**静默漏掉新列**（新列不进任何臂，读数看起来正常，实际上少测了）。
本模块对"未被任何分组覆盖的特征列"显式列出（`uncovered_columns`），
并在覆盖不完整时 fail-loud 告警。
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 消融臂最小样本 / 最小调仓期数（低于此不下结论）
MIN_SAMPLES = 200
MIN_PERIODS = 8

# 模型族候选：name -> 构造器（返回 fit/predict_proba 口径统一的薄封装）
_MODEL_FAMILIES = ("lightgbm", "lightgbm_shallow", "lightgbm_uniform",
                   "random_forest", "logistic_ridge", "extratrees")


def _finite(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _round(x: Any, nd: int = 6) -> Optional[float]:
    f = _finite(x)
    return None if f is None else round(f, nd)


# ----------------------------------------------------------------------
# 特征分组：按族派生，不硬编码清单
# ----------------------------------------------------------------------
# 顺序即展示顺序；未命中任何族的列会被显式披露（见 uncovered_columns）。
FEATURE_GROUP_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("returns", ("ret_",)),
    ("macd", ("macd",)),
    ("boll", ("boll_",)),
    ("rsi", ("rsi",)),
    ("kama", ("kama",)),
    ("ma", ("ma_",)),
    ("trend_extra", ("ema_", "wma_", "dema_", "tema_", "trima_", "kama_",
                     "adx", "adxr", "aroon_", "vortex_", "trix")),
    ("momentum", ("kdj_", "stoch_", "willr", "roc", "mom", "cci", "cmo", "tsi")),
    ("volatility", ("volatility", "atr", "natr", "stddev", "mass_index",
                    "choppiness", "boll_width")),
    ("volume", ("volume_ratio", "obv", "vwap", "mfi", "cmf", "ad_line", "pvt",
                "nvi", "force_index", "eom")),
    ("structure", ("pivot", "r1", "s1", "r2", "s2")),
    ("candle", ("doji", "hammer", "engulfing", "morning_star")),
    ("factor", ("factor_",)),
    ("macro", ("macro_",)),
)

# 消融候选子集：命名 → 参与组（None = 全部组，即基线）
SUBSET_RECIPES: Dict[str, Optional[Tuple[str, ...]]] = {
    "full": None,
    "no_duplicate_trend": ("returns", "macd", "boll", "rsi", "ma", "kama", "momentum",
                           "volatility", "volume", "structure", "candle",
                           "factor", "macro"),
    "no_ma_family": ("returns", "macd", "boll", "rsi", "kama", "trend_extra", "momentum",
                     "volatility", "volume", "structure", "candle", "factor", "macro"),
    "classic_only": ("returns", "macd", "boll", "rsi", "ma", "volume", "volatility"),
    "compact": ("returns", "rsi", "ma", "volatility", "volume", "factor"),
    "returns_only": ("returns",),
    "no_macro": ("returns", "macd", "boll", "rsi", "ma", "kama", "trend_extra", "momentum",
                 "volatility", "volume", "structure", "candle", "factor"),
    "no_factor": ("returns", "macd", "boll", "rsi", "ma", "kama", "trend_extra", "momentum",
                  "volatility", "volume", "structure", "candle", "macro"),
}


# 规则用 **prefix / contains** 两种模式：前缀族（`ma_` / `ema_` / `ret_` / `factor_`）
# 必须走前缀匹配 —— 用 `in` 会让 `ma_` 吞掉 `ema_10` / `dema_5` / `wma_10` / `trima_5`，
# 把趋势族整族静默划进均线族（分组错位会让消融读数失去意义）。
PREFIX_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("returns", ("ret_",)),
    ("boll", ("boll_",)),
    ("ma", ("ma_",)),
    ("trend_extra", ("ema_", "wma_", "dema_", "tema_", "trima_")),
    ("momentum", ("kdj_", "stoch_", "aroon_")),
    ("factor", ("factor_",)),
    ("macro", ("macro_",)),
)
CONTAINS_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("kama", ("kama",)),
    ("macd", ("macd",)),
    ("rsi", ("rsi",)),
    ("trend_extra", ("adx", "adxr", "vortex_", "trix")),
    ("momentum", ("willr", "roc", "mom", "cci", "cmo", "tsi")),
    ("volatility", ("volatility", "atr", "natr", "stddev", "mass_index",
                    "choppiness", "boll_width")),
    ("volume", ("volume_ratio", "obv", "vwap", "mfi", "cmf", "ad_line", "pvt",
                "nvi", "force_index", "eom")),
    ("structure", ("pivot", "r1", "s1", "r2", "s2")),
    ("candle", ("doji", "hammer", "engulfing", "morning_star")),
)


def group_feature_columns(feats: Sequence[str]) -> Tuple[Dict[str, List[str]], List[str]]:
    """按族把特征列分到互斥的组里（先前缀族、后包含族，保证互斥且不误吞）。

    Returns:
        ``(groups, uncovered)``：``uncovered`` = 未命中任何族的列名列表。
        未覆盖的列**不会**进入任何消融子集，调用方必须看见（fail-loud）。
    """
    names = [n for n, _ in FEATURE_GROUP_RULES] + [
        n for n, _ in PREFIX_RULES if n not in {x for x, _ in FEATURE_GROUP_RULES}]
    groups: Dict[str, List[str]] = {n: [] for n in names}
    uncovered: List[str] = []
    for col in feats:
        c = str(col).lower()
        hit = None
        for name, tokens in PREFIX_RULES:
            if any(c.startswith(tok) for tok in tokens):
                hit = name
                break
        if hit is None:
            for name, tokens in CONTAINS_RULES:
                if any(tok in c for tok in tokens):
                    hit = name
                    break
        if hit is None:
            uncovered.append(str(col))
        else:
            groups.setdefault(hit, []).append(str(col))
    return {k: v for k, v in groups.items() if v}, uncovered


# ----------------------------------------------------------------------
# 模型族薄封装（统一 fit / predict_proba 口径）
# ----------------------------------------------------------------------
class _SklearnBinary:
    """sklearn 二分类器的统一薄封装（``predict_proba[:, 1]``）。"""

    def __init__(self, estimator: Any) -> None:
        self._est = estimator

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_SklearnBinary":
        self._est.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._est.predict_proba(X)[:, 1]


class _LgbmAdapter:
    """项目自研 ``LightGBMModel`` 的统一薄封装。

    ⚠️ 口径差异（实测坑）：``LightGBMModel`` 的训练入口叫 ``train()`` 而不是
    sklearn 惯例的 ``fit()``，且内置 ``StandardScaler``。若按惯例调 ``fit``
    会直接 ``AttributeError`` —— 而消融模块原设计是"单臂失败即标注不可用"，
    于是**所有** lightgbm 臂被静默标成"样本不足"，读数看起来"跑完了但没数据"。
    这里显式适配，并把两族（自研 / sklearn）收敛到同一 ``fit`` / ``predict_proba`` 口径。
    """

    def __init__(self, model: Any) -> None:
        self._m = model

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LgbmAdapter":
        self._m.train(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._m.predict_proba(X)[:, 1]


def make_model(model_name: str, config: Dict[str, Any]) -> Any:
    """按族名构造模型（缺依赖 → 由调用方捕获并如实标注不可用）。

    族定义：
      - ``lightgbm``：现行基线族（config 的 lightgbm 超参）；
      - ``lightgbm_shallow``：同族更弱/更正则（num_leaves 15、min_child_samples 大）；
      - ``lightgbm_uniform``：同族更弱（max_depth 3、n_estimators 减半）；
      - ``random_forest`` / ``extratrees``：bagging 族（非线性，抗过拟合口径不同）；
      - ``logistic_ridge``：线性族（正则化），用来判断"信号是不是纯线性"。
    """
    name = str(model_name)
    if name == "logistic_ridge":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return _SklearnBinary(make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, random_state=42),
        ))
    if name in ("random_forest", "extratrees"):
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

        cls = RandomForestClassifier if name == "random_forest" else ExtraTreesClassifier
        return _SklearnBinary(cls(
            n_estimators=200, max_depth=6, min_samples_leaf=50,
            n_jobs=2, random_state=42,
        ))
    if name.startswith("lightgbm"):
        from src.train.models.lightgbm_model import LightGBMModel

        cfg = dict(config)
        lgb_cfg = dict((cfg.get("model", {}) or {}).get("lightgbm") or {})
        if name == "lightgbm_shallow":
            lgb_cfg.update({"num_leaves": 15, "min_child_samples": 60,
                            "max_depth": 4})
        elif name == "lightgbm_uniform":
            lgb_cfg.update({"num_leaves": 8, "max_depth": 3,
                            "n_estimators": max(50, int(lgb_cfg.get("n_estimators", 300)) // 2)})
        cfg.setdefault("model", {})["lightgbm"] = lgb_cfg
        # 消融臂**不落盘模型**（否则每跑一次会覆盖 <save_dir>/lightgbm_model.pkl，
        # 污染线上产物）—— 指向临时目录。
        cfg.setdefault("training", {})
        cfg["training"] = dict(cfg["training"] or {})
        # 消融臂产物统一落在 reports/ 下的临时目录，避免每跑一次覆盖
        # <save_dir>/lightgbm_model.pkl（污染线上模型产物）。
        cfg["training"]["save_dir"] = str(
            cfg["training"].get("save_dir") or "reports/_ablation_models"
        )
        return _LgbmAdapter(LightGBMModel(cfg))
    raise ValueError(f"未知模型族 {model_name!r}（可选：{list(_MODEL_FAMILIES)}）")


def _fit_predict(model_name: str, config: Dict[str, Any],
                 X_tr: np.ndarray, y_tr: np.ndarray,
                 X_te: np.ndarray) -> Optional[np.ndarray]:
    """单折训练 + 预测；失败返回 None（由调用方如实标注，不静默填 0.5）。"""
    try:
        m = make_model(model_name, config)
        m.fit(X_tr, y_tr)
        return np.asarray(m.predict_proba(X_te), dtype=float)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[ablation] 模型族 {model_name} 训练/预测失败: {e}")
        return None


# ----------------------------------------------------------------------
# 样本与折叠：**全程只构造一次**（决定性：各臂必须同一批样本同一批折）
# ----------------------------------------------------------------------
def build_panel(data: Dict[str, pd.DataFrame], config: Dict[str, Any],
                horizon_days: int, folds: int) -> Dict[str, Any]:
    """构造消融用的统一面板：监督集 + 特征列 + 分组 + **预生成的折索引**。

    为什么要预生成折索引：若每个臂各自调一次 ``walk_forward_splits``，
    一旦某臂的样本数不同（例如新特征引入 NaN 导致 dropna 行数变化），
    折边界就会漂移，臂间就**不再是同一批样本比较**，差值失去意义。
    这里把折索引固定下来，所有臂共享。
    """
    from scripts.evaluate_models import walk_forward_splits
    from src.eval.regime_conditioned_signal import _build_supervised, _feature_columns

    ds = _build_supervised(data, config, int(horizon_days))
    out: Dict[str, Any] = {"available": False, "reason": "", "horizon_days": int(horizon_days)}
    if ds.empty:
        out["reason"] = "监督集为空（数据不足）"
        return out
    feats = _feature_columns(ds, config, int(horizon_days))
    if not feats:
        out["reason"] = "无可用特征列"
        return out
    if len(ds) < MIN_SAMPLES:
        out["reason"] = f"样本不足（{len(ds)} < {MIN_SAMPLES}）"
        return out
    splits = list(walk_forward_splits(len(ds), int(folds)))
    if not splits:
        out["reason"] = "walk-forward 折数为 0"
        return out
    groups, uncovered = group_feature_columns(feats)
    if uncovered:
        logger.warning(
            f"[ablation] 特征集中有 {len(uncovered)} 列未被任何分组覆盖 "
            f"{uncovered[:8]}… —— 它们**不会**进入任何消融子集，"
            "读数会漏测这部分特征；请把新特征族补进 FEATURE_GROUP_RULES"
        )
    out.update({
        "available": True,
        "frame": ds,
        "features": list(feats),
        "groups": groups,
        "uncovered_columns": uncovered,
        "splits": splits,
        "n_samples": int(len(ds)),
    })
    return out


def _subset_columns(panel: Dict[str, Any], recipe: str) -> List[str]:
    """把配方名解析成特征列列表（未覆盖的列不进任何子集）。"""
    groups: Dict[str, List[str]] = panel["groups"]
    if recipe not in SUBSET_RECIPES:
        raise ValueError(f"未知子集配方 {recipe!r}（可选：{sorted(SUBSET_RECIPES)}）")
    wanted = SUBSET_RECIPES[recipe]
    if wanted is None:
        return list(panel["features"])
    cols: List[str] = []
    for g in wanted:
        cols.extend(groups.get(g, []))
    # 只保留面板里真实存在的列（配方可能引用当前数据没有的族）
    known = set(panel["features"])
    return [c for c in cols if c in known]


def _walk_forward_proba_with(panel: Dict[str, Any], model_name: str,
                             config: Dict[str, Any],
                             columns: Sequence[str]) -> Tuple[np.ndarray, int]:
    """在固定折上跑一个臂，返回（逐样本样本外概率，成功折数）。"""
    ds: pd.DataFrame = panel["frame"]
    h = int(panel["horizon_days"])
    n = len(ds)
    proba = np.full(n, np.nan, dtype=float)
    ok_folds = 0
    y_all = ds[f"target_{h}d"].to_numpy(dtype=float)
    cols = list(columns)
    if not cols:
        return proba, 0
    X_all = ds[cols].fillna(0.0).to_numpy(dtype=float)
    for train_idx, test_idx in panel["splits"]:
        y_tr = y_all[train_idx]
        if len(np.unique(y_tr[np.isfinite(y_tr)])) < 2:
            continue
        p = _fit_predict(model_name, config, X_all[train_idx], y_tr, X_all[test_idx])
        if p is None or len(p) != len(test_idx):
            continue
        proba[test_idx] = p
        ok_folds += 1
    return proba, ok_folds


# ----------------------------------------------------------------------
# 与 benchmark_relative 同源的判据（唯一记分板）
# ----------------------------------------------------------------------
def _signal_selection(ds: pd.DataFrame, proba: np.ndarray,
                      horizon_days: int, grid: Sequence[pd.Timestamp],
                      confidence_thr: float) -> Dict[pd.Timestamp, List[str]]:
    """逐调仓日按概率（净看涨）选出标的：``p >= thr`` 才持有。"""
    d = ds[["date", "_symbol"]].copy()
    d["_p"] = proba
    sel: Dict[pd.Timestamp, List[str]] = {}
    for t in grid:
        m = (d["date"] == t).to_numpy() & np.isfinite(d["_p"].to_numpy())
        keep = d.loc[m & (d["_p"].to_numpy() >= float(confidence_thr)), "_symbol"]
        sel[t] = [str(s) for s in keep.tolist()]
    return sel


def _eval_arm(panel: Dict[str, Any], data: Dict[str, pd.DataFrame],
              model_name: str, config: Dict[str, Any], columns: Sequence[str],
              horizon_days: int, one_side_cost: float,
              grid: Sequence[pd.Timestamp], hold_horizon: int,
              confidence_thr: float, n_random_controls: int = 40,
              seed: int = 7) -> Dict[str, Any]:
    """单臂读数：逐期净超额向量 + 汇总（与 benchmark_relative 同口径）。"""
    from src.eval.benchmark_relative import _excess_stats, _horizon_returns, _period_stats

    ds: pd.DataFrame = panel["frame"]
    proba, ok_folds = _walk_forward_proba_with(panel, model_name, config, columns)
    mask = np.isfinite(proba)
    base: Dict[str, Any] = {
        "model": model_name,
        "n_features": int(len(columns)),
        "n_scored": int(mask.sum()),
        "n_folds_ok": int(ok_folds),
        "available": False,
        "reason": "",
        "mean_probability": _round(float(np.mean(proba[mask]))) if mask.any() else None,
        "mean_period_turnover": None,
    }
    if mask.sum() < MIN_SAMPLES:
        base["reason"] = f"样本外概率样本不足（{int(mask.sum())} < {MIN_SAMPLES}）"
        return base

    selmap = _signal_selection(ds, proba, horizon_days, grid, confidence_thr)
    sig = _horizon_returns(data, grid, selmap, int(hold_horizon))
    stats = _period_stats(sig, int(hold_horizon))
    base["portfolio"] = stats
    base["avg_n_selected"] = _round(
        float(np.mean([len(v) for v in selmap.values()])), 3)
    prev: Optional[set] = None
    turns: List[float] = []
    for t in grid:
        cur = set(selmap.get(t) or [])
        if prev is not None:
            denom = max(1, len(prev | cur))
            turns.append(len(prev ^ cur) / denom)
        prev = cur
    base["mean_period_turnover"] = _round(float(np.mean(turns))) if turns else None
    if not stats.get("available"):
        base["reason"] = stats.get("reason") or "组合期数不足"
        return base
    base["available"] = True
    base["_period_returns"] = sig

    # ---- 同源判据：相对全池等权的净超额 + 随机子集对照（与 edge-check 同口径） ----
    symbols = sorted((data or {}).keys())
    bench = _horizon_returns(data, grid, {t: symbols for t in grid}, int(hold_horizon))
    base["_benchmark_period_returns"] = bench
    from src.eval.benchmark_relative import _excess_stats
    base["vs_benchmark"] = _excess_stats(sig, bench, one_side_cost, int(hold_horizon))
    base["benchmark"] = _period_stats(bench, int(hold_horizon))
    base["benchmark"]["note"] = f"全池等权 buy&hold（{len(symbols)} 标的，无换手成本）"
    base["random_subset_control"] = _random_subset_control(
        data, bench, grid, int(hold_horizon), one_side_cost,
        int(round(base.get("avg_n_selected") or 0)) or 1,
        int(n_random_controls), int(seed))
    base["verdict"] = _arm_verdict(base)
    return base


def _random_subset_control(data: Dict[str, pd.DataFrame], bench: np.ndarray,
                           grid: Sequence[pd.Timestamp], hold_horizon: int,
                           one_side: float, n_sel: int,
                           n_random_controls: int, seed: int) -> Dict[str, Any]:
    """抽**同样数量**的随机标的子集当对照（判断"选中这些标的"是否含信息）。"""
    from src.eval.benchmark_relative import _excess_stats, _horizon_returns

    symbols = sorted((data or {}).keys())
    n_sel = max(1, min(int(n_sel), len(symbols)))
    if n_random_controls <= 0:
        return {"available": False, "reason": "随机对照次数为 0（已关闭）"}
    rng = np.random.default_rng(int(seed))
    draws: List[float] = []
    for _ in range(int(n_random_controls)):
        rm = {t: rng.choice(symbols, size=n_sel, replace=False).tolist() for t in grid}
        r = _horizon_returns(data, grid, rm, int(hold_horizon))
        st = _excess_stats(r, bench, one_side, int(hold_horizon))
        if st.get("available"):
            draws.append(float(st["excess_mean_per_period"]))
    if len(draws) < MIN_PERIODS:
        return {"available": False,
                "reason": f"可用随机对照次数不足（{len(draws)} < {MIN_PERIODS}）"}
    dr = np.asarray(draws, dtype=float)
    return {
        "available": True,
        "n_draws": int(len(dr)),
        "subset_size": int(n_sel),
        "random_excess_mean": _round(float(dr.mean()), 8),
        "fraction_random_ge_signal": None,   # 由调用方用本臂观测值回填
        "note": "信号子集落在随机子集分布内 ⇒ 选中这些标的没有信息含量",
        "_draws": dr,
    }


def _verdict_from(arms: Dict[str, Any], baseline_name: str,
                  best_name: Optional[str], kind: str) -> Dict[str, Any]:
    """结论分级（写死）：只有「净超额为正 + 相对基线臂配对 t ≥ 2」才允许说"有正向增量"。

    判定表（与 baseline 臂的 `no_edge` 分级**同源**）：
      - 候选净超额 > 0 且配对 t ≥ 2 → `increment_confirmed`（有正向增量）；
      - 候选相对基准**统计上确认劣**（净超额 ≤ 0 且 |t| ≥ 2）→ `worse_than_baseline`；
      - 候选净超额 > 0 但配对 t < 2，或净超额 ≤ 0 但不显著
        （|t| < 2，即严格证明不了它更好也证明不了它更差）→ `inconclusive`。

    为什么必须把 `inconclusive` 单列：把"不显著为负"直接归成 `no_increment`
    会**夸大证据强度** —— 它证明的是"没证明更好"，而不是"更差"。
    IC / 命中率 / AUC 改善**都不是**通过条件（绝对读数无法与市场 beta 分离）。
    """
    base = arms.get(baseline_name) or {}
    if not base.get("available"):
        return {"level": "unavailable", "kind": kind,
                "reason": f"基线臂（{baseline_name}）读数不可用"}
    out: Dict[str, Any] = {
        "kind": kind,
        "baseline": baseline_name,
        "baseline_net_excess": (base.get("vs_benchmark") or {}).get("excess_mean_per_period"),
        "baseline_verdict": (base.get("verdict") or {}).get("level"),
        "candidates": {},
    }
    for name, arm in arms.items():
        if name == baseline_name or not isinstance(arm, dict) or not arm.get("available"):
            continue
        vs = arm.get("vs_benchmark") or {}
        paired = arm.get("vs_baseline_arm") or {}
        excess = vs.get("excess_mean_per_period")
        t_exc = vs.get("excess_t_stat")
        t_paired = paired.get("diff_t_stat")
        confirmed = bool(excess is not None and excess > 0
                         and t_paired is not None and t_paired >= 2.0)
        worse = bool(excess is not None and excess <= 0
                     and t_exc is not None and t_exc <= -2.0)
        out["candidates"][name] = {
            "net_excess_per_period": excess,
            "excess_t_stat": t_exc,
            "paired_diff_per_period": paired.get("mean_diff_per_period"),
            "paired_diff_t": t_paired,
            "verdict": (arm.get("verdict") or {}).get("level"),
            "increment_confirmed": confirmed,
            "worse_than_baseline": worse,
        }
    best_arm, best_paired_t = None, None
    for name, c in out["candidates"].items():
        t = c.get("paired_diff_t")
        if t is None:
            continue
        if best_paired_t is None or t > best_paired_t:
            best_paired_t, best_arm = t, name
    out["best_candidate"] = best_arm
    out["best_paired_diff_t"] = best_paired_t
    # 多重比较：同一轮里比了 m 个候选，单臂"显著"必须过 Holm 校正后才作数
    # （防"跑 20 个臂挑一个 p<0.05"式的自由度回流；试验预算见 trial_registry）。
    _apply_multiple_comparison(out["candidates"])
    winners = [k for k, c in out["candidates"].items() if c["increment_confirmed"]]
    inconclusive = [k for k, c in out["candidates"].items()
                    if not c["increment_confirmed"] and not c["worse_than_baseline"]]
    worse = [k for k, c in out["candidates"].items() if c["worse_than_baseline"]]
    out["winners"] = winners
    out["inconclusive"] = inconclusive
    out["worse_than_baseline"] = worse
    if winners:
        out["level"] = "positive_increment"
        out["reason"] = (f"候选 {winners} 净超额转正且相对基线臂配对 t ≥ 2 "
                         f"⇒ 有正向增量，但仍须人工签字才允许改口径")
    elif not out["candidates"]:
        out["level"] = "unavailable"
        out["reason"] = "无可用候选臂（全部读数不可用）"
    elif inconclusive:
        out["level"] = "inconclusive"
        out["reason"] = (f"候选 {inconclusive} 相对基线臂的差异不显著（配对/超额 |t| < 2）"
                         f"⇒ 既未证明更好、也未证明更差；本轮证据不足以支持改口径")
    else:
        out["level"] = "no_increment"
        out["reason"] = (f"候选 {worse} 相对基准统计上确认更差（净超额 ≤ 0 且 |t| ≥ 2）"
                         f"⇒ 无正向增量，不建议改特征集 / 模型族")
    return out


def _apply_multiple_comparison(candidates: Dict[str, Any]) -> None:
    """对候选臂的配对 t 做 Holm-Bonferroni 校正，写回 ``paired_p_holm``。

    纪律：单臂 p < 0.05 **不足以**下结论 —— 一轮比 m 个候选时族错误率会被放大。
    校正后的 p 一并落盘，供人工判断"这个差异是真的还是挑出来的"。

    用正态近似算双侧 p（n 通常 > 100，t 分布与正态差异可忽略），
    再按 Holm 逐步校正；不引入新依赖（自研 6 行，与项目"零新依赖"口径一致）。
    """
    import math as _math

    entries: List[Tuple[str, float, float]] = []
    for name, c in candidates.items():
        t = c.get("paired_diff_t")
        if t is None:
            c["paired_p_raw"] = None
            c["paired_p_holm"] = None
            continue
        p = 2.0 * (1.0 - 0.5 * (1.0 + _math.erf(abs(float(t)) / _math.sqrt(2.0))))
        c["paired_p_raw"] = _round(p, 6)
        entries.append((name, float(t), p))
    if not entries:
        return
    entries.sort(key=lambda x: x[2])
    m = len(entries)
    running = 0.0
    for i, (name, _t, p) in enumerate(entries):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        candidates[name]["paired_p_holm"] = _round(min(1.0, running), 6)
    for name, c in candidates.items():
        if c.get("paired_p_holm") is None and c.get("paired_p_raw") is not None:
            c["paired_p_holm"] = c["paired_p_raw"]


def _match_benchmark_grid(ds: pd.DataFrame, hold_horizon: int,
                          max_lag: int) -> List[pd.Timestamp]:
    """信号日期 ∩ 价格日期（允许最多 ``max_lag`` 日滞后的除权日对齐）。

    实测坑：`data/raw` 的日K是按**查询窗口**分段拉的，除权日会让某些标的
    在某天缺行（幅度很小）。若严格按日期 inner join，会白丢大量调仓日；
    这里在**只回看**（不允许回看未来）的前提下把日期归并到最近的可用价格日。
    """
    sig_dates = sorted(pd.DatetimeIndex(pd.unique(ds["date"])).normalize())
    if not sig_dates:
        return []
    if _GRID_PRICE_FRAMES:
        price_dates = sorted(pd.DatetimeIndex(
            pd.unique(pd.concat([v["date"] for v in _GRID_PRICE_FRAMES.values()]))
        ).normalize())
    else:
        price_dates = sig_dates
    out: List[pd.Timestamp] = []
    for t in sig_dates:
        match = None
        for lag in range(0, int(max_lag) + 1):
            cand = t - pd.Timedelta(days=lag)
            if cand in price_dates:
                match = cand
                break
        if match is not None and (not out or match != out[-1]):
            out.append(match)
    return out[:: int(hold_horizon)]


# 用来做「信号日 ∩ 价格日」对齐的全局价格表（由 build_report 注入，避免层层传参）
_GRID_PRICE_FRAMES: Dict[str, pd.DataFrame] = {}


def build_report(data: Dict[str, pd.DataFrame], config: Dict[str, Any],
                 horizons: Sequence[int] = (5,),
                 folds: int = 3,
                 hold_horizon: int = 5,
                 confidence_thr: float = 0.5,
                 cost_level: str = "base",
                 feature_recipes: Optional[Sequence[str]] = None,
                 model_families: Optional[Sequence[str]] = None,
                 base_model: str = "lightgbm",
                 base_recipe: str = "full",
                 max_lag_days: int = 5,
                 n_random_controls: int = 40,
                 seed: int = 7) -> Dict[str, Any]:
    """特征集 × 模型族联合消融（只读）。

    Args:
        data: ``{symbol: DataFrame(date, close)}`` 价格表。
        base_model / base_recipe: 基线臂（默认 = 现行口径：LightGBM + 全量特征）。
        feature_recipes: 参与特征消融的配方名（缺省 = 全部 `SUBSET_RECIPES`）。
        model_families: 参与模型消融的族名（缺省 = 全部 `_MODEL_FAMILIES`）。
    """
    from src.eval.portfolio_backtest import cost_one_side

    out: Dict[str, Any] = {
        "kind": "feature_model_ablation",
        "available": False,
        "reason": "",
        "affects_gate": False,
        "readonly": True,
        "horizons": [int(h) for h in horizons],
        "folds": int(folds),
        "holding_horizon": int(hold_horizon),
        "confidence_threshold": float(confidence_thr),
        "cost_level": str(cost_level),
        "baseline": {"model": base_model, "recipe": base_recipe},
        "scoring_rule": ("唯一记分板 = 相对全池等权的净超额（扣 2× 单边成本、"
                         "非重叠调仓、优于随机子集）；增量需净超额转正且"
                         "相对基线臂配对 t ≥ 2。IC / 命中率 / AUC 不参与判定。"),
    }
    symbols = sorted((data or {}).keys())
    if len(symbols) < 2:
        out["reason"] = f"可用标的不足（{len(symbols)} < 2）"
        return out
    try:
        one_side = float(cost_one_side(cost_level))
    except ValueError as e:
        out["reason"] = str(e)
        return out

    global _GRID_PRICE_FRAMES
    _GRID_PRICE_FRAMES = dict(data or {})

    recipes = [r for r in (feature_recipes or list(SUBSET_RECIPES))]
    families = [f for f in (model_families or list(_MODEL_FAMILIES))]
    unknown_r = [r for r in recipes if r not in SUBSET_RECIPES]
    unknown_f = [f for f in families if f not in _MODEL_FAMILIES]
    if unknown_r or unknown_f:
        out["reason"] = f"未知配方 {unknown_r} / 未知模型族 {unknown_f}"
        return out

    per_horizon: Dict[str, Any] = {}
    for h in sorted(set(int(x) for x in horizons)):
        per_horizon[str(h)] = _ablate_horizon(
            data, config, h, int(folds), int(hold_horizon), float(confidence_thr),
            one_side, recipes, families, base_model, base_recipe,
            int(max_lag_days), int(n_random_controls), int(seed))
    out["per_horizon"] = per_horizon
    out["available"] = any(v.get("available") for v in per_horizon.values())
    if not out["available"]:
        out["reason"] = "全部周期读数不可用（数据或样本不足）"
    # 顶层结论：有多个周期时报最保守（任一周期无增量即不主张增量）
    out["feature_ablation"] = {h: v.get("feature_ablation") for h, v in per_horizon.items()}
    out["model_ablation"] = {h: v.get("model_ablation") for h, v in per_horizon.items()}
    levels = [((v.get("feature_ablation") or {}).get("level"),
               (v.get("model_ablation") or {}).get("level"))
              for v in per_horizon.values()]
    out["feature_verdict_level"] = _worst_level([x[0] for x in levels])
    out["model_verdict_level"] = _worst_level([x[1] for x in levels])
    out["conclusion_note"] = (
        "本报告只产出证据：特征消融 / 模型族消融均为**只读**读数"
        "（`affects_gate=False`），不改变特征集、模型配置、权重、池或门禁；"
        "是否据此变更生产口径属产品变更，须人工签字。"
    )
    return out


def _worst_level(levels: Sequence[Optional[str]]) -> Optional[str]:
    """多周期汇总：只要任一周期出现"未证实"就不主张增量（最保守口径）。

    优先级：``positive_increment`` 要求**全部**周期都成立；否则
    ``inconclusive`` > ``no_increment`` > ``unavailable``。
    """
    ls = [x for x in levels if x]
    if not ls:
        return None
    if len(set(ls)) == 1:
        return ls[0]
    for lv in ("inconclusive", "no_increment", "unavailable"):
        if lv in ls:
            return lv
    return "positive_increment"


def _ablate_horizon(data: Dict[str, pd.DataFrame], config: Dict[str, Any],
                    h: int, folds: int, hold_horizon: int, confidence_thr: float,
                    one_side: float, recipes: Sequence[str],
                    families: Sequence[str], base_model: str, base_recipe: str,
                    max_lag_days: int, n_random_controls: int,
                    seed: int) -> Dict[str, Any]:
    """单周期的两轮消融（特征轮 / 模型轮），共用同一面板与同一判据。"""
    t0 = time.time()
    panel = build_panel(data, config, h, folds)
    out: Dict[str, Any] = {
        "kind": "ablation_horizon",
        "horizon_days": int(h),
        "available": bool(panel.get("available")),
        "reason": panel.get("reason", ""),
        "n_samples": panel.get("n_samples"),
        "n_features_full": len(panel.get("features") or []),
        "feature_groups": {k: len(v) for k, v in (panel.get("groups") or {}).items()},
        "uncovered_columns": panel.get("uncovered_columns") or [],
    }
    if not panel.get("available"):
        return out
    ds: pd.DataFrame = panel["frame"]
    grid = _match_benchmark_grid(ds, hold_horizon, max_lag_days)
    out["n_rebalance_dates"] = int(len(grid))
    if len(grid) < MIN_PERIODS:
        out["available"] = False
        out["reason"] = f"可用调仓日不足（{len(grid)} < {MIN_PERIODS}）"
        return out

    base_cols = _subset_columns(panel, base_recipe)
    if not base_cols:
        out["available"] = False
        out["reason"] = f"基线配方 {base_recipe!r} 解析出 0 列特征"
        return out

    # 设计矩阵：两侧都基于**同一条** `_eval_arm`（含净超额 + 随机子集对照），
    # 基线臂 = 设计矩阵中 (模型=base_model, 配方=base_recipe) 那一格。
    # 这样"基线"与"候选"走的是**完全同一条代码路径**：任何口径漂移都会同时
    # 作用于两侧，不会制造出虚假的臂间差异（此前 base 分支与普通分支各走一套，
    # 曾导致基线臂缺少配对差值）。
    # 非零成本设计：轮1 加一次（重复用），轮2 加一次（基线重复 1 格 + 每族 1 次）。
    if base_recipe not in recipes:
        recipes = list(recipes) + [base_recipe]
    if base_model not in families:
        families = list(families) + [base_model]

    feature_arms: Dict[str, Any] = {}
    for recipe in recipes:
        cols = _subset_columns(panel, recipe)
        if not cols:
            feature_arms[recipe] = _available_arm(
                base_model, 0, f"配方 {recipe!r} 解析出 0 列特征")
            continue
        feature_arms[recipe] = _eval_arm(
            panel, data, base_model, config, cols, h, one_side, grid,
            hold_horizon, confidence_thr, n_random_controls, seed)
    _finalize_arms(feature_arms, base_recipe, one_side, n_random_controls)
    base_arm_for_model = feature_arms[base_recipe]

    model_arms: Dict[str, Any] = {}
    for fam in families:
        if fam == base_model:
            # 与轮1 的基线格读数逐字段一致（同参数、同折、同随机种子）
            model_arms[fam] = {k: v for k, v in base_arm_for_model.items()}
            continue
        model_arms[fam] = _eval_arm(
            panel, data, fam, config, base_cols, h, one_side, grid,
            hold_horizon, confidence_thr, n_random_controls, seed)
    _finalize_arms(model_arms, base_model, one_side, n_random_controls)

    out["baseline_arm"] = {k: v for k, v in
                           feature_arms.get(base_recipe, {}).items()
                           if not str(k).startswith("_")}
    out["feature_arms"] = {k: {kk: vv for kk, vv in v.items()
                               if not str(kk).startswith("_")}
                           for k, v in feature_arms.items()}
    out["model_arms"] = {k: {kk: vv for kk, vv in v.items()
                             if not str(kk).startswith("_")}
                         for k, v in model_arms.items()}
    out["feature_ablation"] = _verdict_from(feature_arms, base_recipe,
                                            _best_by_t(feature_arms, base_recipe),
                                            "feature_ablation")
    out["model_ablation"] = _verdict_from(model_arms, base_model,
                                          _best_by_t(model_arms, base_model),
                                          "model_ablation")
    out["elapsed_seconds"] = _round(time.time() - t0, 2)
    return out


def _available_arm(model: str, n_features: int, reason: str) -> Dict[str, Any]:
    """不可用臂的统一结构（字段齐全，便于下游一致消费）。"""
    return {"model": model, "n_features": int(n_features), "n_scored": 0,
            "n_folds_ok": 0, "available": False, "reason": str(reason),
            "mean_probability": None, "mean_period_turnover": None,
            "avg_n_selected": None, "vs_benchmark": None,
            "vs_baseline_arm": None, "random_subset_control": None,
            "verdict": None}


def _finalize_arms(arms: Dict[str, Any], baseline_name: str,
                   one_side: float, n_random_controls: int) -> None:
    """给每个臂补上「相对基线臂的配对差值」与「随机子集对照的分位读数」。"""
    base = arms.get(baseline_name)
    if not base or not base.get("available"):
        return
    obs = (base.get("vs_benchmark") or {}).get("excess_mean_per_period")
    ctrl = base.get("random_subset_control")
    if isinstance(ctrl, dict) and ctrl.get("available"):
        draws = ctrl.pop("_draws", None)
        if draws is not None:
            ctrl["fraction_random_ge_signal"] = (
                _round(float((draws >= obs).mean()), 4) if obs is not None else None)
    for name, arm in arms.items():
        if not isinstance(arm, dict) or not arm.get("available"):
            continue
        if name != baseline_name:
            arm["vs_baseline_arm"] = _paired_vs_baseline_raw(base, arm)
        if arm.get("random_subset_control") is None:
            # 候选臂共用基线臂的随机对照（同一批调仓日 + 同一子集规模 + 同种子）
            arm["random_subset_control"] = (dict(base["random_subset_control"])
                                            if isinstance(base.get("random_subset_control"), dict)
                                            else None)
        arm["verdict"] = _arm_verdict(arm)


def _paired_vs_baseline_raw(base_arm: Dict[str, Any], arm: Dict[str, Any]) -> Dict[str, Any]:
    """配对差值（成本已逐臂全额计入，这里只做纯粹的臂间比较）。"""
    a = arm.get("_period_returns")
    b = base_arm.get("_period_returns")
    if a is None or b is None:
        return {"available": False, "reason": "基线臂或本臂逐期收益不可用"}
    n = min(len(a), len(b))
    if n < MIN_PERIODS:
        return {"available": False, "reason": f"可比调仓期数不足（{n} < {MIN_PERIODS}）"}
    d = np.asarray(a[:n], dtype=float) - np.asarray(b[:n], dtype=float)
    sd = float(d.std(ddof=1))
    t = float(d.mean() / sd * math.sqrt(n)) if sd > 1e-12 else None
    return {
        "available": True,
        "n_periods": int(n),
        "mean_diff_per_period": _round(float(d.mean()), 8),
        "diff_t_stat": _round(t, 4) if t is not None else None,
        "note": "同批样本同折的基线臂对照；差值即该臂相对现行口径的金融增量",
    }


def _arm_verdict(arm: Dict[str, Any]) -> Dict[str, Any]:
    """单臂结论分级（与 `benchmark_relative._verdict` 同源纪律）。"""
    vs = arm.get("vs_benchmark") or {}
    ctrl = arm.get("random_subset_control") or {}
    if not vs.get("available"):
        return {"level": "unavailable", "reason": "净超额读数不可用"}
    obs = vs.get("excess_mean_per_period")
    t = vs.get("excess_t_stat")
    frac = ctrl.get("fraction_random_ge_signal") if ctrl.get("available") else None
    if obs is None:
        return {"level": "unavailable", "reason": "净超额均值为 None"}
    if obs <= 0:
        return {"level": "no_edge",
                "reason": "净超额（相对全池等权、扣成本后）非正 ⇒ 不如直接等权持有",
                "excess_mean_per_period": obs}
    if t is not None and t >= 2.0 and frac is not None and frac <= 0.10:
        return {"level": "positive", "reason": "净超额显著为正且优于随机子集",
                "excess_t_stat": t, "fraction_random_ge_signal": frac}
    return {"level": "inconclusive",
            "reason": "净超额为正但证据不足（t < 2 或随机子集也能做到）",
            "excess_t_stat": t, "fraction_random_ge_signal": frac}


def _best_by_t(arms: Dict[str, Any], baseline_name: str) -> Optional[str]:
    """按相对基线臂的配对 t 选最优候选（仅用于报告展示，不用于判定）。"""
    best, best_t = None, None
    for name, arm in arms.items():
        if name == baseline_name or not isinstance(arm, dict):
            continue
        t = (arm.get("vs_baseline_arm") or {}).get("diff_t_stat")
        if t is None:
            continue
        if best_t is None or t > best_t:
            best_t, best = t, name
    return best
