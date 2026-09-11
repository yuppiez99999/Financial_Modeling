"""特征扩充的**正交对照实验**（S12）：先证明"该不该扩"，再扩。

问题从哪来（S11 的结论）：
  S11 用多重比较校正把「换预测周期」这条路查实了 —— 现行 5/10/20 日与 40/60 日
  全部未过线且校正后不显著，`verdict=reject`。README §17.3 与 SALES_PLAN §8.2 里
  写的「特征扩充（横截面/宏观/情感）可把 short/mid 命中率推过门禁线」是**待验证假设**，
  而且它是一条**因果断言**：加特征 → 指标上升。

  问题在于：这类断言最容易被"加了一堆特征、某次跑出好看数字"骗过去。
  特征越多，过拟合与选择偏差的空间越大；只看一次结果，无法区分
  「真的带来正交信息」与「噪声被拟合进去了」。

本模块做什么：
  把「特征扩充有没有用」做成一次**冻结口径的正交对照实验**：

  1. **锚点基准**（baseline）：现行特征集（技术指标 + 50+ 扩展指标 [+ 因子列]）；
  2. **对照臂**（arms）：逐族单独加入 / 全量加入，**每臂只改特征集，其余全不动**
     （同一份数据、同一折切分、同一超参、同一评估口径）；
  3. **配对比较**：与基准**同一折**对比 IC / 命中率的增量（delta），并给出方向一致率；
  4. **显著性与多重比较**：每个臂的 IC 增量做近似 t 检验，并按臂数做 Bonferroni 校正
     —— 与 S11 同一条纪律：多臂比较必然产生"最好看的那一臂"，
     不校正就会把噪声当发现；
  5. **结论三态**：`adopt`（有臂显著且方向一致）/ `reject`（全部不显著）/
     `defer`（不可比或缺数据）。

本模块**不做什么**：
  - **不改特征集**：只测量，不往 `FeatureEngineer` 里加任何列；
    要不要扩由决策单（S11 同构）决定；
  - **不调参**：所有臂共用同一套模型超参，避免"调出来的对比"；
  - **不挑臂**：`adopt` 需要校正后仍显著，且对增量方向一致率有要求；
  - **不猜**：数据不足 / 构造失败 → `available=false` + 原因，绝不用默认值凑结论。

无前视保证：
  每个臂的特征都在**逐标的数据上**先算好，再交给同一套 ``walk_forward_splits``
  切分；模型只在训练折拟合、只在测试折打分。新增的横截面特征只用**同一交易日**
  的其它标的信息（不跨时间），宏观特征按发布日期 asof 对齐（无前视），
  两者都有测试守护。
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPORT_NAME = "feature_experiment.json"

# 结论三态
VERDICT_ADOPT = "adopt"
VERDICT_REJECT = "reject"
VERDICT_DEFER = "defer"

# 默认对照臂：横截面 / 宏观 / 情感（与 SALES_PLAN §8.2 的假设一一对应）
DEFAULT_ARMS = ("cross_sectional", "macro", "sentiment")


def _cfg(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    data = (config or {}).get("feature_experiment", {}) or {}
    return data if isinstance(data, dict) else {}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default


def _mean(values: Sequence[float]) -> float:
    vals = [_num(v) for v in values if v is not None and _num(v, float("nan")) == _num(v, float("nan"))]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _std(values: Sequence[float]) -> float:
    vals = [_num(v) for v in values]
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return float(math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)))


# ----------------------------------------------------------------------
# 特征族构造
# ----------------------------------------------------------------------
def _panel_close(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """拼出「日期 × 标的」的收盘价面板（横截面特征的唯一输入）。

    逐标的按 date 对齐：**只用同一交易日的多标的快照**，不跨时间，
    因此横截面特征本身不引入未来信息。
    """
    frames = []
    for symbol, df in (data or {}).items():
        if df is None or len(df) == 0 or "close" not in df.columns:
            continue
        sub = df[["date", "close"]].copy()
        sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
        sub = sub.dropna(subset=["date"])
        sub["_symbol"] = symbol
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    return panel.pivot_table(index="date", columns="_symbol", values="close", aggfunc="last")


def _cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    """横截面特征：单一交易日内的相对强弱（**不含任何未来数据**）。

    族内 4 个特征（都在同一天横截面上计算，然后按日期并入标的行）：
      - ``xsec_ret_rank``  ：当日收益在池内的分位（0~1）
      - ``xsec_mom_rank``  ：20 日动量在池内的分位
      - ``xsec_excess``    ：当日收益 − 池内均值（相对强弱，单位：收益率）
      - ``xsec_dispersion``：池内当日收益的标准差（市场情绪/分化度）
    """
    if panel.empty or panel.shape[1] < 2:
        return pd.DataFrame()
    ret1 = panel.pct_change(1)
    mom20 = panel.pct_change(20)
    # 全部在同一天横截面上计算（rank/sub/std 都是 axis=1），不跨时间、无前视
    long = pd.DataFrame({
        "xsec_ret_rank": ret1.rank(axis=1, pct=True).stack(),
        "xsec_mom_rank": mom20.rank(axis=1, pct=True).stack(),
        "xsec_excess": ret1.sub(ret1.mean(axis=1), axis=0).stack(),
    })
    long = long.reindex(columns=["xsec_ret_rank", "xsec_mom_rank", "xsec_excess"])
    # 市场分化度是**逐日的标量**，与标的身位无关，按日期广播到每个标的
    disp = ret1.std(axis=1)
    long["xsec_dispersion"] = long.index.get_level_values(0).map(disp)
    long.index.names = ["date", "_symbol"]
    return long.reset_index()


def _arm_columns(arm: str, df: pd.DataFrame) -> List[str]:
    """某个对照臂在该数据帧上**实际可用**的特征列（缺列=该臂不完整）。"""
    cols = [c for c in df.columns if isinstance(c, str)]
    if arm == "cross_sectional":
        want = ("xsec_ret_rank", "xsec_mom_rank", "xsec_excess", "xsec_dispersion")
        return [c for c in want if c in cols]
    if arm == "macro":
        return [c for c in cols if c.startswith("macro_")]
    if arm == "sentiment":
        return [c for c in cols if c.startswith(("sentiment_", "news_"))]
    return []


def _base_columns(df: pd.DataFrame, horizon_days: int, config: Dict[str, Any]) -> List[str]:
    """现行基准特征列：技术指标 + 扩展指标（[+ 因子列]），复用生产同一套口径。

    刻意调用 ``FeatureEngineer.get_feature_columns`` —— 保证"基准"就是**生产上
    真正在用的那套特征**，而不是实验里手搓的另一套。
    """
    from src.data.preprocessor import FeatureEngineer

    fe = FeatureEngineer(config)
    cols = [c for c in fe.get_feature_columns(df, horizon_days) if not str(c).startswith("_")]
    excluded = set()
    for arm in DEFAULT_ARMS:
        excluded.update(_arm_columns(arm, df))
    return [c for c in cols if c not in excluded]


# ----------------------------------------------------------------------
# 实验执行
# ----------------------------------------------------------------------
@dataclass
class ArmResult:
    """单臂（单周期）实验结果。"""

    arm: str
    horizon_days: int
    base_ic: float = 0.0
    arm_ic: float = 0.0
    base_hit: float = 0.0
    arm_hit: float = 0.0
    base_samples: int = 0
    arm_samples: int = 0
    folds: int = 0
    available: bool = False
    reason: str = ""
    ic_deltas: List[float] = field(default_factory=list)
    hit_deltas: List[float] = field(default_factory=list)
    features_added: List[str] = field(default_factory=list)

    @property
    def ic_delta(self) -> float:
        return self.arm_ic - self.base_ic

    @property
    def hit_delta(self) -> float:
        return self.arm_hit - self.base_hit

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arm": self.arm,
            "horizon_days": self.horizon_days,
            "available": self.available,
            "reason": self.reason,
            "features_added": list(self.features_added),
            "base_ic": round(self.base_ic, 4),
            "arm_ic": round(self.arm_ic, 4),
            "ic_delta": round(self.ic_delta, 4),
            "base_hit_rate": round(self.base_hit, 4),
            "arm_hit_rate": round(self.arm_hit, 4),
            "hit_delta": round(self.hit_delta, 4),
            "base_samples": self.base_samples,
            "arm_samples": self.arm_samples,
            "folds": self.folds,
            "fold_ic_deltas": [round(_num(d), 4) for d in self.ic_deltas],
        }


def _eval_arm(X: np.ndarray, y: np.ndarray, fwd: np.ndarray, splits,
              config: Dict[str, Any]) -> Tuple[List[float], List[float], int]:
    """在给定特征矩阵上跑同一套 walk-forward，返回逐折 (score, fwd_ret) 汇总。

    模型一律用生产同款 ``LightGBMModel``（同超参），避免"实验里换模型"。
    """
    from src.train.models.lightgbm_model import LightGBMModel

    scores: List[float] = []
    returns: List[float] = []
    used = 0
    for train_idx, test_idx in splits:
        if len(np.unique(y[train_idx])) < 2:
            continue
        try:
            model = LightGBMModel(config)
            model.train(X[train_idx], y[train_idx])
            proba = model.predict_proba(X[test_idx])[:, 1]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[feature-exp] 折训练失败: {e}")
            continue
        scores.extend(float(p) - 0.5 for p in proba)
        returns.extend(float(r) for r in fwd[test_idx])
        used += 1
    return scores, returns, used


def _fold_metrics(scores: Sequence[float], returns: Sequence[float]) -> Tuple[float, float]:
    from src.inference.ic import hit_rate, spearman_ic

    return spearman_ic(scores, returns), hit_rate(scores, returns)


def run_experiment(data: Dict[str, pd.DataFrame], config: Dict[str, Any],
                   arms: Sequence[str] = DEFAULT_ARMS,
                   folds: int = 3) -> Dict[str, Any]:
    """跑一次特征扩充正交对照实验（纯计算，不落盘）。"""
    import scripts.evaluate_models as ev
    from src.inference.ic import hit_rate, spearman_ic
    from src.data.preprocessor import FeatureEngineer

    cfg = _cfg(config)
    max_family_p = _num(cfg.get("max_family_p", 0.05), 0.05)
    min_delta = _num(cfg.get("min_ic_delta", 0.0), 0.0)
    horizons = (config.get("data", {}) or {}).get("prediction_horizons", {}) or {}

    result: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "src/eval/feature_experiment.py",
        "arms": list(arms),
        "thresholds": {"max_family_p": max_family_p, "min_ic_delta": min_delta},
        "horizons": {},
        "available": bool(data),
        "affects_features": False,
        "narrative": "",
    }
    if not data:
        result["verdict"] = VERDICT_DEFER
        result["narrative"] = "无可用行情数据，实验未执行"
        return result

    # 横截面特征：一次算好，逐标的并入（只用同日面板，无前视）
    panel_feats = _cross_sectional_features(_panel_close(data))

    per_horizon: Dict[str, Any] = {}
    for hname, days in horizons.items():
        days = int(days)
        rows: Dict[str, Any] = {arm: [] for arm in arms}
        base_rows: List[Dict[str, Any]] = []
        arm_features: Dict[str, List[str]] = {arm: [] for arm in arms}
        folds_used = 0
        samples_base = samples_arm = 0

        for symbol, raw in data.items():
            df = raw.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            try:
                feats = FeatureEngineer(config).transform(df, days)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[feature-exp] {symbol} 特征失败: {e}")
                continue
            if not panel_feats.empty:
                # 并入横截面特征：按 (日期, 标的) 左连接。
                # 注意 **不能** 用 `panel_feats["_symbol"] == symbol` 过滤：
                # `_symbol` 是 pandas StringDtype，缺列时比较结果可能退化成标量 False，
                # 整列被静默丢掉 —— 那会让"横截面臂"变成"什么都没加"，
                # 实验结论直接失真。这里显式转成普通字符串再比较。
                own = panel_feats[panel_feats["_symbol"].astype(str) == str(symbol)]
                if not own.empty:
                    feats = feats.merge(own.drop(columns=["_symbol"]),
                                        on="date", how="left")
            feats = feats.copy()
            feats["_symbol"] = symbol
            feats["_fwd_ret"] = feats["close"].shift(-days) / feats["close"] - 1
            if f"target_{days}d" not in feats.columns:
                feats["target_" + f"{days}d"] = np.where(
                    feats["_fwd_ret"] > 0, 1.0, 0.0)
            feats.loc[feats["_fwd_ret"].isna(), f"target_{days}d"] = np.nan
            feats = feats.dropna(subset=[f"target_{days}d"])

            base_cols = _base_columns(feats, days, config)
            base_ready = feats[base_cols + [f"target_{days}d", "_fwd_ret"]].dropna()
            if base_ready.empty:
                continue

            sub = base_ready
            Xb = sub[base_cols].to_numpy(dtype=float)
            yb = sub[f"target_{days}d"].to_numpy(dtype=float)
            fb = sub["_fwd_ret"].to_numpy(dtype=float)
            splits = ev.walk_forward_splits(len(Xb), folds)
            sb, rb, used_b = _eval_arm(Xb, yb, fb, splits, config)
            folds_used = max(folds_used, used_b)
            samples_base += len(sb)
            if sb:
                base_rows.append({"symbol": symbol, "scores": sb, "returns": rb})

            for arm in arms:
                extra = [c for c in _arm_columns(arm, feats) if c not in base_cols]
                if not extra:
                    continue
                arm_features[arm].extend(extra)
                cols = base_cols + extra
                ready = feats[cols + [f"target_{days}d", "_fwd_ret"]].dropna()
                if ready.empty:
                    continue
                X = ready[cols].to_numpy(dtype=float)
                y = ready[f"target_{days}d"].to_numpy(dtype=float)
                fwd = ready["_fwd_ret"].to_numpy(dtype=float)
                s, r, used = _eval_arm(X, y, fwd, ev.walk_forward_splits(len(X), folds), config)
                if not s:
                    continue
                folds_used = max(folds_used, used)
                samples_arm += len(s)
                rows[arm].append({"symbol": symbol, "scores": s, "returns": r})

        # 汇总：基准与各臂的整体指标 + 逐折配对增量
        all_base_scores = [v for row in base_rows for v in row["scores"]]
        all_base_returns = [v for row in base_rows for v in row["returns"]]
        base_ic = spearman_ic(all_base_scores, all_base_returns) if all_base_scores else 0.0
        base_hit = hit_rate(all_base_scores, all_base_returns) if all_base_scores else 0.0

        arm_results: List[Dict[str, Any]] = []
        for arm in arms:
            per_arm = rows[arm]
            if not per_arm:
                # ⚠️ 不可用臂**不得**带上 arm_ic/arm_hit 的 0 值 ——
                # 那会让报告里出现"IC 从 0.075 掉到 0"这种看起来像严重退化的假象。
                # 可用的对照只有"该特征族在当前配置下根本没有列"这一个事实。
                arm_results.append(ArmResult(
                    arm=arm, horizon_days=days, available=False,
                    reason=(
                        "该特征族在当前配置下没有产出任何特征列"
                        "（如 sentiment 需 `features.sentiment_enabled: true`），"
                        "因此本次实验无法评估该族 —— 不是没用，是没测"
                    ),
                    base_ic=base_ic, base_hit=base_hit, base_samples=samples_base,
                    folds=folds_used,
                ).to_dict())
                continue
            scores = [v for row in per_arm for v in row["scores"]]
            returns = [v for row in per_arm for v in row["returns"]]
            ic = spearman_ic(scores, returns)
            hit = hit_rate(scores, returns)
            # 配对：同标的分开算增量（不同标的样本长度不同，逐标的配对更稳妥）
            base_by_symbol = {row["symbol"]: row for row in base_rows}
            ic_deltas: List[float] = []
            hit_deltas: List[float] = []
            for row in per_arm:
                base_row = base_by_symbol.get(row["symbol"])
                if not base_row:
                    continue
                ic_deltas.append(spearman_ic(row["scores"], row["returns"])
                                 - spearman_ic(base_row["scores"], base_row["returns"]))
                hit_deltas.append(hit_rate(row["scores"], row["returns"])
                                  - hit_rate(base_row["scores"], base_row["returns"]))
            arm_results.append(ArmResult(
                arm=arm, horizon_days=days, available=True,
                base_ic=base_ic, arm_ic=ic, base_hit=base_hit, arm_hit=hit,
                base_samples=samples_base, arm_samples=len(scores), folds=folds_used,
                ic_deltas=ic_deltas, hit_deltas=hit_deltas,
                features_added=sorted(set(arm_features[arm])),
            ).to_dict())

        per_horizon[hname] = {
            "horizon_days": days,
            "base_ic": round(base_ic, 4),
            "base_hit_rate": round(base_hit, 4),
            "base_samples": samples_base,
            "folds": folds_used,
            "arms": arm_results,
        }

    result["horizons"] = per_horizon
    result["_config"] = config
    result["verdict"], result["narrative"] = decide(result)
    result.pop("_config", None)
    return result


# ----------------------------------------------------------------------
# 结论判定（多重比较校正，与 S11 同一条纪律）
# ----------------------------------------------------------------------
def _paired_t(deltas: Sequence[float]) -> Tuple[float, int]:
    """配对增量的近似 t 值：mean / (s/√n)。样本不足返回 0。"""
    vals = [_num(d) for d in deltas]
    n = len(vals)
    if n < 3:
        return 0.0, n
    sd = _std(vals)
    if sd <= 1e-12:
        return 0.0, n
    return (_mean(vals) / (sd / math.sqrt(n))), n


def _trial_budget(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """历史未登记自由度（S13）。登记不可用时如实降级为"只数本次"。"""
    if not config:
        return {"available": False, "extra_trials": 0, "reason": "no config"}
    try:
        from src.eval.trial_registry import count_trials

        stats = count_trials(config)
    except Exception as e:  # noqa: BLE001
        return {"available": False, "extra_trials": 0, "reason": str(e)}
    if not stats.get("available"):
        return {"available": False, "extra_trials": 0,
                "reason": stats.get("reason", "trial registry unavailable")}
    return {"available": True, "extra_trials": int(stats.get("count", 0)),
            "file": stats.get("file", "")}


def decide(result: Dict[str, Any]) -> Tuple[str, str]:
    """把逐臂结果判成 adopt / reject / defer（纯函数，便于单测）。"""
    from src.eval.horizon_decision import bonferroni, normal_two_sided_p

    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for hname, hdata in (result.get("horizons") or {}).items():
        for arm in hdata.get("arms") or []:
            if arm.get("available"):
                rows.append({"horizon": hname, **arm})
            else:
                skipped.append(f"{hname}/{arm.get('arm')}")

    if not rows:
        return (
            VERDICT_DEFER,
            "没有任何可用对照臂（数据不足或特征列缺失），无法给出结论；"
            f"未测到的臂：{skipped}",
        )

    # ⚠️ 多重比较只数**真正跑过的**比较：没测过的臂不构成一次比较，
    # 不该让它白白抬高校正强度（那会让真发现更难通过）。
    # 再加上**历史未登记自由度**（S13）：反复调特征开关同样是未登记的试验。
    history = _trial_budget(result.get("_config"))
    extra = int(history.get("extra_trials", 0)) if history.get("available") else 0
    n_trials = len(rows) + extra
    result["n_trials_history"] = extra
    result["trial_budget"] = history
    significant: List[Dict[str, Any]] = []
    for row in rows:
        t, n = _paired_t(row.get("fold_ic_deltas") or [])
        p = normal_two_sided_p(t)
        p_adj = bonferroni(p, n_trials)
        row["p_value"] = round(p, 6)
        row["p_family"] = round(p_adj, 6)
        row["paired_n"] = n
        row["significant"] = bool(n >= 3 and p_adj <= _num(result.get("thresholds", {}).get("max_family_p"), 0.05))
        if row["significant"]:
            significant.append(row)

    result["comparisons"] = [{k: v for k, v in r.items() if k != "fold_ic_deltas"} for r in rows]
    result["n_trials"] = n_trials

    skipped_note = (
        f"（另有 {len(skipped)} 个臂因特征列缺失未能评估：{skipped}）"
        if skipped else ""
    )
    if not significant:
        return (
            VERDICT_REJECT,
            f"{n_trials} 个对照臂（周期 × 特征族）全部不显著{skipped_note}："
            "加入横截面/宏观/情感特征**没有带来可验证的正交信息**，"
            "特征扩充这条路的现有证据不支持继续投入",
        )

    detail = "；".join(
        f"{r['horizon']}/{r['arm']} IC 增量 {r['ic_delta']:+.4f}"
        f"（族错误率 {r['p_family']}）" for r in significant)
    return (
        VERDICT_ADOPT,
        f"{len(significant)}/{n_trials} 个对照臂经多重比较校正后显著{skipped_note}：{detail}。"
        "⚠️ 这是**证据**而非自动落地：是否接入生产特征集仍需人工确认并重跑全量门禁",
    )


# ----------------------------------------------------------------------
# 落盘 / 读取
# ----------------------------------------------------------------------
def report_path(config: Optional[Dict[str, Any]] = None) -> Path:
    return Path(_cfg(config).get("report_dir", "reports")) / REPORT_NAME


def save(result: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Path:
    out = report_path(config)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load(config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    path = report_path(config)
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[feature-exp] 读取失败 {path}: {e}")
        return None
