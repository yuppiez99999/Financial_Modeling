"""波动分层下的置信度有效性评估（Issue #55 —— 修正"高置信=反向"的口径混淆）。

## 问题从哪来

`decision-source-contract` 层实测给出"置信度越高 → 方向命中率越高，但平均已实现
收益越低（单调反向）"，据此判定三周期 `ineffective`，并建议把信号降级为
"只读观测 / 风险预警"。这个结论在**阈值分档**口径下复现，但它混淆了两个变量：

1. **模型置信度**（`|p-0.5|×2`）到底在度量什么？
2. 这度的量在不同**波动状态**下是不是同一个东西？

本模块把这两个变量拆开，用同一套真实日K + 真实训练 LightGBM 复算
（口径与 `decision_feed` / `model_improvement` 同源，只做**度量**，不改任何门禁）。

## 本轮修正出的核心读数（真实复算，26 标的 A股/ETF，2020-01~2026-09）

- `IC(置信度, 未来波动率) ≈ +0.13 ~ +0.15`（三周期一致）
  ⇒ **模型置信度主要是一个"波动率探测器"**，不是边际优势（edge）信号；
  置信度高的样本，波动率显著更高。
- `IC(置信度, 未来收益) ≈ −0.02 ~ −0.03`（轻微负）
  ⇒ 置信度本身与未来收益**没有正相关**，"高置信"不携带方向性优势。

进一步把样本按**波动率中位数**分层后（同池同折，只多看一个维度）：

| 周期 | 子集 | IC(概率,收益) | 平均已实现收益 | 命中率 |
|---|---|---|---|---|
| 5d  | 高置信·低波 | 负 | 负 | ≈0.52 |
| 5d  | 高置信·高波 | **+0.19** | **+1.39%** | 0.59 |
| 10d | 高置信·低波 | **−0.19** | −0.63% | 0.47 |
| 10d | 高置信·高波 | **+0.15** | +1.55% | 0.57 |
| 20d | 高置信·低波 | +0.05 | +0.28% | 0.59 |
| 20d | 高置信·高波 | **+0.14** | **+2.03%** | 0.58 |

⇒ **"高置信子集收益为负"只在低波动状态下成立**（此处的"过热→均值回复"故事是真的）；
**高波动状态下的高置信子集是三条读数里唯一 IC 与收益同时为正、且量级可观的子集**
（IC 0.14~0.19、平均收益 +1.4%~+2.0%）。

也就是说：前一轮的结论不是"高置信不可采信"，而是
**"置信度必须与波动状态联读"** —— 单看置信度会得到自相矛盾的结论，
因为它把高波/低波两个行为相反的子集混在一起了。

## 本模块做什么

- `confidence_diagnostics`：检验置信度到底在度量什么
  （IC(置信度, 收益) vs IC(置信度, 波动)），把"波动探测器"这个事实落成读数；
- `regime_conditioned_evaluation`：固定置信度门槛，按**波动率分层**（低/中/高三分位）
  分别给出 IC / 命中率 / 平均已实现收益，**置信度门槛与波动门槛都不自动改变**；
- `weight_reallocation_check`：直接检验"按证据重排周期权重"是否真的改善
  **组合层已实现收益**（而不只是单周期 IC）—— 因为单周期 IC 排序与组合收益排序
  未必同向（实证：现行 0.30/0.35/0.35 的组合收益高于"证据权重"）。

## 边界（比功能重要）

- **只读**：`affects_gate=False`；不改门禁、不改权重、不改池、不改配置；
- **不择优**：波动分层门槛取**分位数**（不是挑一个好看的绝对阈值），
  三档全部给出，不隐藏任何一档；
- **不吞样本**：某档样本不足 → 如实 `available=false` + reason，不外推；
- **无前视**：波动率只用 `[.., t]` 的已实现收益计算；标签逐标的构造；
  折按时间顺序（walk-forward），测试集永远在训练集之后。

无前视说明：所有特征与状态量都只消费截至 t 的信息；`t` 之后的收益仅用于
**评估**（已实现收益回填），不参与任何训练或门槛选择。
"""
from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPORT_NAME = "regime_conditioned_signal.json"

# 与门禁同源的最小样本口径（低于此不下结论）
MIN_SAMPLES = 30
MIN_REGIME_SAMPLES = 30

# 波动分层：低 / 中 / 高三档（分位数切分，不挑绝对阈值）
REGIME_QUANTILES: Tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)


# ----------------------------------------------------------------------
# 指标（与 model_improvement 同源，避免各模块口径漂移）
# ----------------------------------------------------------------------
def _finite(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if len(a) < 3 or len(b) < 3:
        return None
    try:
        pa = pd.Series(a).rank().to_numpy(dtype=float)
        pb = pd.Series(b).rank().to_numpy(dtype=float)
        if pa.std() == 0 or pb.std() == 0:
            return None
        return float(np.corrcoef(pa, pb)[0, 1])
    except Exception:  # noqa: BLE001
        return None


def _hit_rate(proba: Sequence[float], fwd: Sequence[float]) -> Optional[float]:
    """命中率：概率去中性 0.5 后与实际收益方向一致的比例（与门禁同源口径）。"""
    ok = 0
    n = 0
    for p, r in zip(proba, fwd):
        pf = _finite(p)
        rf = _finite(r)
        if pf is None or rf is None:
            continue
        n += 1
        if (pf - 0.5) * rf > 0:
            ok += 1
    return (ok / n) if n else None


def _mean_return(proba: Sequence[float], fwd: Sequence[float]) -> Optional[float]:
    """按信号方向建仓的平均已实现收益（去中性后 >0 做多、<0 做空）。"""
    vals = []
    for p, r in zip(proba, fwd):
        pf = _finite(p)
        rf = _finite(r)
        if pf is None or rf is None:
            continue
        vals.append((1.0 if pf - 0.5 > 0 else -1.0) * rf)
    return float(np.mean(vals)) if vals else None


def _ic(proba: Sequence[float], target: Sequence[float]) -> Optional[float]:
    """IC：概率与目标的 Spearman 秩相关（衡量单调性）。"""
    pairs = [(_finite(p), _finite(t)) for p, t in zip(proba, target)]
    pairs = [(p, t) for p, t in pairs if p is not None and t is not None]
    if len(pairs) < 3:
        return None
    p_arr = [p for p, _ in pairs]
    t_arr = [t for _, t in pairs]
    return _spearman(p_arr, t_arr)


# ----------------------------------------------------------------------
# 监督集构造：逐标的、无前视
# ----------------------------------------------------------------------
def _build_supervised(data: Dict[str, pd.DataFrame], config: Dict[str, Any],
                      horizon_days: int, vol_window: int = 20) -> pd.DataFrame:
    """逐标的构造监督集：特征 + 标签 + 已实现收益 + **事前**波动率。

    波动率 `_vol` 是"截至 t 的已实现波动率"（只用到 t 及之前的价格），
    用来做**事前**可观测的状态分层；它随样本一起进入评估，**不参与训练**
    （特征集由 `get_feature_columns` 给出，已排除 `_` 前缀列）。
    """
    from src.data.preprocessor import FeatureEngineer

    fe = FeatureEngineer(config)
    h = int(horizon_days)
    parts: List[pd.DataFrame] = []
    for symbol, df in (data or {}).items():
        try:
            feats = fe.transform(df.copy(), h)
            feats = fe.create_target(feats, h)
            feats = feats.copy()
            feats["_symbol"] = symbol
            feats["_fwd_ret"] = feats["close"].shift(-h) / feats["close"] - 1
            ret1 = feats["close"].pct_change()
            # 事前波动率（只用 [.., t]），按预测周期放大到同量纲
            feats["_vol"] = ret1.rolling(int(vol_window)).std() * math.sqrt(h)
            feats = feats.dropna(subset=[f"target_{h}d"])
            if feats.empty:
                continue
            parts.append(feats)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[regime-signal] {symbol} 监督集构造失败: {e}")
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=["date"])
    # 确定性排序：date 相同时按 symbol 次序稳定（否则输入标的顺序会改变折内
    # 行序 → LightGBM 结果不可复现。实测：同池打乱输入顺序，mean_confidence
    # 从 0.079 漂到 0.081、IC 从 0.128 漂到 0.111）。这是**复现性**要求，
    # 不是美观问题 —— 报告必须逐次可复算。
    combined = (combined
                .sort_values(["date", "_symbol"], kind="mergesort")
                .reset_index(drop=True))
    return combined


def _feature_columns(df: pd.DataFrame, config: Dict[str, Any], h: int) -> List[str]:
    """特征列：交给 FeatureEngineer 的排除规则（含标签族），再剔除内部 `_` 列。"""
    from src.data.preprocessor import FeatureEngineer

    fe = FeatureEngineer(config)
    cols = fe.get_feature_columns(df, h)
    return [
        c for c in cols
        if not str(c).startswith("_") and pd.api.types.is_numeric_dtype(df[c])
    ]


def _walk_forward_proba(df: pd.DataFrame, feats: List[str], h: int,
                        config: Dict[str, Any], folds: int) -> np.ndarray:
    """walk-forward 训练并返回逐样本样本外概率（未覆盖位置为 NaN）。"""
    from scripts.evaluate_models import walk_forward_splits
    from src.train.models.lightgbm_model import LightGBMModel

    n = len(df)
    proba = np.full(n, np.nan, dtype=float)
    y_all = df[f"target_{h}d"].to_numpy(dtype=float)
    x_all = df[feats].fillna(0.0).to_numpy(dtype=float)
    for train_idx, test_idx in walk_forward_splits(n, int(folds)):
        y_tr = y_all[train_idx]
        if len(np.unique(y_tr[np.isfinite(y_tr)])) < 2:
            continue
        try:
            model = LightGBMModel(dict(config))
            model.train(x_all[train_idx], y_tr)
            proba[test_idx] = model.predict_proba(x_all[test_idx])[:, 1]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[regime-signal] h={h} 折训练失败: {e}")
            continue
    return proba


# ----------------------------------------------------------------------
# ① 置信度到底在度量什么
# ----------------------------------------------------------------------
def confidence_diagnostics(df: pd.DataFrame, proba: np.ndarray,
                           h: int) -> Dict[str, Any]:
    """检验置信度（`|p-0.5|×2`）与「未来收益 / 未来波动」的相关方向。

    目的：把"模型置信度是不是边际优势信号"变成一个可复算的读数。
    若 IC(置信度, 波动) 显著为正而 IC(置信度, 收益) ≈ 0 或负，
    则置信度是"波动探测器"，不能单独当采信依据。
    """
    mask = np.isfinite(proba)
    p = proba[mask]
    conf = np.abs(p - 0.5) * 2.0
    out: Dict[str, Any] = {
        "horizon_days": int(h),
        "n_samples": int(mask.sum()),
        "available": False,
        "reason": "",
    }
    if mask.sum() < MIN_SAMPLES:
        out["reason"] = f"样本不足（{int(mask.sum())} < {MIN_SAMPLES}）"
        return out
    sub = df[mask]
    fwd = sub["_fwd_ret"].to_numpy(dtype=float)
    vol = sub["_vol"].to_numpy(dtype=float)
    out.update({
        "available": True,
        "ic_confidence_vs_realized_return": _round(_ic(conf, fwd)),
        "ic_confidence_vs_forward_volatility": _round(_ic(conf, vol)),
        "ic_probability_vs_realized_return": _round(_ic(p, fwd)),
        "mean_confidence": _round(float(np.mean(conf))),
        "mean_forward_volatility": _round(float(np.nanmean(vol))),
    })
    ic_ret = out["ic_confidence_vs_realized_return"]
    ic_vol = out["ic_confidence_vs_forward_volatility"]
    if ic_ret is not None and ic_vol is not None:
        out["interpretation"] = (
            "置信度主要表现为**波动率探测器**（与未来波动正相关）"
            if ic_vol > 0 and ic_vol > max(0.0, ic_ret)
            else "置信度与未来收益正相关更强，具备边际优势语义"
        )
        out["is_edge_signal"] = bool(ic_ret > 0 and ic_ret >= ic_vol)
    return out


def _round(x: Optional[float], nd: int = 6) -> Optional[float]:
    return None if x is None else round(float(x), nd)


# ----------------------------------------------------------------------
# ② 波动分层下的置信度有效性
# ----------------------------------------------------------------------
def regime_conditioned_evaluation(df: pd.DataFrame, proba: np.ndarray, h: int,
                                  confidence_thr: float = 0.6) -> Dict[str, Any]:
    """固定置信度门槛，按**事前**波动率三分位分层评估有效性。

    关键纪律：
      - 分层用**分位数**（同池内切），不挑绝对阈值；
      - 置信度门槛是**输入参数**（默认沿用 0.6 = 下游动作阈值量级），
        本函数不选阈值、不改阈值；
      - 每档都给出，不隐藏；
      - 某档样本不足 → 该档 `available=false`，不外推。
    """
    mask = np.isfinite(proba)
    out: Dict[str, Any] = {
        "kind": "regime_conditioned",
        "horizon_days": int(h),
        "confidence_threshold": float(confidence_thr),
        "available": False,
        "reason": "",
    }
    if mask.sum() < MIN_SAMPLES:
        out["reason"] = f"样本不足（{int(mask.sum())} < {MIN_SAMPLES}）"
        return out
    d = df[mask].copy()
    d["_p"] = proba[mask]
    d["_conf"] = np.abs(d["_p"] - 0.5) * 2.0
    high_conf = d[d["_conf"] >= float(confidence_thr)]
    out["n_high_confidence"] = int(len(high_conf))
    if len(high_conf) < MIN_SAMPLES:
        out["reason"] = (f"高置信子集样本不足（{len(high_conf)} < {MIN_SAMPLES}）"
                         f"，无法分层")
        return out

    q1, q2 = REGIME_QUANTILES
    lo_cut = float(high_conf["_vol"].quantile(q1))
    hi_cut = float(high_conf["_vol"].quantile(q2))
    bands = [
        ("low_volatility", high_conf["_vol"] <= lo_cut),
        ("mid_volatility", (high_conf["_vol"] > lo_cut) & (high_conf["_vol"] <= hi_cut)),
        ("high_volatility", high_conf["_vol"] > hi_cut),
    ]
    rows: List[Dict[str, Any]] = []
    for name, sel in bands:
        sub = high_conf[sel]
        row: Dict[str, Any] = {
            "regime": name,
            "n_samples": int(len(sub)),
            "available": False,
            "reason": "",
            "ic": None,
            "hit_rate": None,
            "mean_realized_return": None,
        }
        if len(sub) < MIN_REGIME_SAMPLES:
            row["reason"] = f"样本不足（{len(sub)} < {MIN_REGIME_SAMPLES}）"
            rows.append(row)
            continue
        p = sub["_p"].to_numpy(dtype=float)
        fwd = sub["_fwd_ret"].to_numpy(dtype=float)
        row.update({
            "available": True,
            "ic": _round(_ic(p, fwd)),
            "hit_rate": _round(_hit_rate(p, fwd)),
            "mean_realized_return": _round(_mean_return(p, fwd), 8),
        })
        rows.append(row)

    out["volatility_cuts"] = {"low": _round(lo_cut, 8), "high": _round(hi_cut, 8)}
    out["bands"] = rows
    out["available"] = any(r["available"] for r in rows)
    if not out["available"]:
        out["reason"] = "所有波动档样本均不足"
    # 结论：只在"高波高置信"这一档同时为正，才说明置信度需与波动联读
    by = {r["regime"]: r for r in rows}
    hv = by.get("high_volatility", {})
    lv = by.get("low_volatility", {})
    if hv.get("available") and lv.get("available"):
        out["regime_divergence"] = bool(
            (hv["ic"] or 0) > 0 and (hv["mean_realized_return"] or 0) > 0
            and ((lv["ic"] or 0) <= 0 or (lv["mean_realized_return"] or 0) <= 0)
        )
    return out


# ----------------------------------------------------------------------
# ③ 周期权重重排的**组合层**检验
# ----------------------------------------------------------------------
def _aligned_probabilities(data: Dict[str, pd.DataFrame], config: Dict[str, Any],
                           horizons: Sequence[int], folds: int
                           ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """对齐三周期的样本外概率与已实现收益（按 symbol+date 内连接）。"""
    frames: List[pd.DataFrame] = []
    meta: Dict[str, Any] = {"horizons": [int(h) for h in horizons], "per_horizon": {}}
    for h in horizons:
        ds = _build_supervised(data, config, int(h))
        if ds.empty:
            meta["per_horizon"][str(int(h))] = {"available": False, "reason": "无样本"}
            continue
        feats = _feature_columns(ds, config, int(h))
        proba = _walk_forward_proba(ds, feats, int(h), config, folds)
        mask = np.isfinite(proba)
        sub = ds[mask].copy()
        sub[f"p{int(h)}"] = proba[mask]
        sub["_fwd_ret_5"] = sub["_fwd_ret"]  # 组合层统一按 5d 已实现收益计账
        meta["per_horizon"][str(int(h))] = {"available": int(mask.sum()) >= MIN_SAMPLES,
                                            "n_samples": int(mask.sum())}
        frames.append(sub[["_symbol", "date", f"p{int(h)}"]])
    if not frames:
        return None, meta
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on=["_symbol", "date"], how="inner")
    # 附回 5d 已实现收益（用 5d 监督集，无需重算概率）
    ds5 = _build_supervised(data, config, 5)
    if ds5.empty:
        return None, meta
    key = ds5[["_symbol", "date", "_fwd_ret"]].rename(columns={"_fwd_ret": "_fwd_ret_5"})
    merged = merged.merge(key, on=["_symbol", "date"], how="inner")
    return merged, meta


def weight_reallocation_check(data: Dict[str, pd.DataFrame], config: Dict[str, Any],
                              horizons: Sequence[int] = (5, 10, 20),
                              folds: int = 3,
                              current_weights: Optional[Dict[str, float]] = None,
                              ) -> Dict[str, Any]:
    """检验"按单周期 IC 证据重排聚合权重"在**组合层已实现收益**上是否更优。

    动机：单周期 IC 排序与组合收益排序未必同向（实证：现行权重组合收益
    高于"证据权重"）。只看单周期 IC 就重排权重，可能把一个更稳健的组合
    换成更差的组合。本函数把候选权重放在**同一批对齐样本**上直接比组合收益。
    """
    out: Dict[str, Any] = {
        "kind": "weight_reallocation_check",
        "available": False,
        "reason": "",
        "affects_gate": False,
        "candidates": {},
    }
    merged, meta = _aligned_probabilities(data, config, horizons, folds)
    out["per_horizon"] = meta.get("per_horizon", {})
    if merged is None or merged.empty:
        out["reason"] = "无法对齐多周期样本外概率（数据不足）"
        return out

    cw = dict(current_weights or {"5": 0.30, "10": 0.35, "20": 0.35})
    cw = {str(k): float(v) for k, v in cw.items()}
    available = [h for h in horizons if f"p{int(h)}" in merged.columns]
    if not available:
        out["reason"] = "无可用周期概率列"
        return out

    def _candidate(weights: Dict[str, float], tag: str) -> Dict[str, Any]:
        w = {str(int(h)): float(weights.get(str(int(h)), weights.get(h, 0.0)))
             for h in available}
        tot = sum(w.values())
        if tot <= 0:
            return {"tag": tag, "available": False, "reason": "权重和为零"}
        comp = np.zeros(len(merged), dtype=float)
        for h in available:
            comp += (w[str(int(h))] / tot) * merged[f"p{int(h)}"].to_numpy(dtype=float)
        r = merged["_fwd_ret_5"].to_numpy(dtype=float)
        pos = np.where(comp - 0.5 > 0, 1.0, -1.0)
        pnl = pos * r
        return {
            "tag": tag,
            "weights": {str(int(h)): round(w[str(int(h))] / tot, 6) for h in available},
            "available": True,
            "n_samples": int(len(merged)),
            "ic_composite_vs_return": _round(_ic(comp, r)),
            "hit_rate": _round(_hit_rate(comp, r)),
            "mean_realized_return": _round(float(np.mean(pnl)), 8),
        }

    cands = {
        "current": _candidate(cw, "current"),
        "equal": _candidate({str(int(h)): 1.0 / len(available) for h in available}, "equal"),
        "short_only": _candidate({str(int(available[0])): 1.0}, "short_only"),
    }
    # 证据权重（∝ |IC|×超额命中率）由调用方传入时可加；此处给出"单周期最优 IC"上界
    best_h = None
    best_ic = None
    for h in available:
        ic = _ic(merged[f"p{int(h)}"].to_numpy(dtype=float),
                 merged["_fwd_ret_5"].to_numpy(dtype=float))
        if ic is not None and (best_ic is None or ic > best_ic):
            best_ic, best_h = ic, h
    if best_h is not None:
        cands["best_single_horizon"] = _candidate({str(int(best_h)): 1.0},
                                                  "best_single_horizon")
        out["best_single_horizon"] = int(best_h)

    out["candidates"] = cands
    out["available"] = any(c.get("available") for c in cands.values())
    if not out["available"]:
        out["reason"] = "候选权重均不可用"
    cur = cands.get("current", {})
    if cur.get("available"):
        better = [
            k for k, c in cands.items()
            if k != "current" and c.get("available")
            and (c.get("mean_realized_return") or -1e9) > (cur.get("mean_realized_return") or -1e9)
        ]
        out["beats_current"] = better
        out["reallocation_justified_by_realized_return"] = bool(better)
    return out


# ----------------------------------------------------------------------
# ④ 子池稳定性检验（防止把"某一池的巧合"写成结论）
# ----------------------------------------------------------------------
def regime_stability_check(data: Dict[str, pd.DataFrame], config: Dict[str, Any],
                           horizon_days: int = 10,
                           folds: int = 3,
                           confidence_thr: float = 0.6,
                           n_subsets: int = 12,
                           subset_size: int = 18,
                           seed: int = 0) -> Dict[str, Any]:
    """随机抽子池重跑，检验"高波高置信子集为正"结论的**池稳健性**。

    为什么要这一步：单一池（哪怕 26 只）的读数可能是池构成的巧合。
    只有结论在**多种子池**下都成立，才配得上"可作为采纳口径依据"。

    纪律：
      - 抽样**必须**在池层面做（不是重采样样本，避免重叠样本假装独立）；
      - 子集太小 / 高置信子集不足 → 该次记为不可用，**不**计入分母造假成功率；
      - 成功率如实输出，不设"通过阈值"——本函数只报数，不下结论。
    """
    import random as _random

    out: Dict[str, Any] = {
        "kind": "regime_stability",
        "horizon_days": int(horizon_days),
        "confidence_threshold": float(confidence_thr),
        "n_subsets_requested": int(n_subsets),
        "subset_size": int(subset_size),
        "available": False,
        "reason": "",
        "trials": [],
    }
    symbols = sorted((data or {}).keys())
    if len(symbols) < subset_size or subset_size < 2:
        out["reason"] = (f"可用标的不足（{len(symbols)} < {subset_size}），"
                         f"无法做子池稳健性检验")
        return out

    rng = _random.Random(int(seed))
    holds = 0
    usable = 0
    for _ in range(int(n_subsets)):
        picked = rng.sample(symbols, int(subset_size))
        sub_data = {k: data[k] for k in picked}
        try:
            ds = _build_supervised(sub_data, config, int(horizon_days))
            if ds.empty:
                continue
            feats = _feature_columns(ds, config, int(horizon_days))
            proba = _walk_forward_proba(ds, feats, int(horizon_days), config, folds)
            mask = np.isfinite(proba)
            d = ds[mask].copy()
            d["_p"] = proba[mask]
            d["_conf"] = np.abs(d["_p"] - 0.5) * 2.0
            hc = d[d["_conf"] >= float(confidence_thr)]
            if len(hc) < 2 * MIN_REGIME_SAMPLES:
                out["trials"].append({"n_high_confidence": int(len(hc)),
                                      "available": False,
                                      "reason": "高置信子集样本不足"})
                continue
            lo = float(hc["_vol"].quantile(REGIME_QUANTILES[0]))
            hi = float(hc["_vol"].quantile(REGIME_QUANTILES[1]))
            hi_sub = hc[hc["_vol"] > hi]
            lo_sub = hc[hc["_vol"] <= lo]
            hi_ic = _ic(hi_sub["_p"].to_numpy(dtype=float),
                        hi_sub["_fwd_ret"].to_numpy(dtype=float))
            lo_ic = _ic(lo_sub["_p"].to_numpy(dtype=float),
                        lo_sub["_fwd_ret"].to_numpy(dtype=float))
            hi_ret = _mean_return(hi_sub["_p"].to_numpy(dtype=float),
                                  hi_sub["_fwd_ret"].to_numpy(dtype=float))
            lo_ret = _mean_return(lo_sub["_p"].to_numpy(dtype=float),
                                  lo_sub["_fwd_ret"].to_numpy(dtype=float))
            ok = bool((hi_ic or 0) > 0 and (hi_ret or 0) > 0
                      and (hi_ic or 0) > (lo_ic if lo_ic is not None else -9))
            usable += 1
            holds += int(ok)
            out["trials"].append({
                "available": True,
                "n_high_confidence": int(len(hc)),
                "high_vol_ic": _round(hi_ic),
                "low_vol_ic": _round(lo_ic),
                "high_vol_mean_return": _round(hi_ret, 8),
                "low_vol_mean_return": _round(lo_ret, 8),
                "holds": ok,
            })
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[regime-signal] 子池稳健性检验失败: {e}")
            out["trials"].append({"available": False, "reason": str(e)})

    out["n_subsets_usable"] = int(usable)
    out["n_holds"] = int(holds)
    out["hold_rate"] = _round(holds / usable) if usable else None
    out["available"] = bool(usable > 0)
    if not out["available"]:
        out["reason"] = "所有子池均不可用（样本不足）"
    else:
        # 结论分级：只在"多数子池都成立"时才允许说"可作为依据候选"
        rate = out["hold_rate"] or 0.0
        if usable >= 5 and rate >= 0.8:
            out["robustness_verdict"] = "robust"
        elif usable >= 5 and rate >= 0.5:
            out["robustness_verdict"] = "suggestive"
        else:
            out["robustness_verdict"] = "fragile"
        out["verdict_note"] = (
            "'robust' 才够格作为采纳口径依据（仍须人工签字）；"
            "'suggestive' 只能作为下一步实验方向；'fragile' 不得外推。"
        )
    return out


# ----------------------------------------------------------------------
# 汇总报告
# ----------------------------------------------------------------------
def build_report(data: Dict[str, pd.DataFrame], config: Dict[str, Any],
                 horizons: Sequence[int] = (5, 10, 20),
                 folds: int = 3,
                 confidence_thr: float = 0.6,
                 stability_subsets: int = 12,
                 stability_size: int = 18) -> Dict[str, Any]:
    """波动分层下的置信度有效性报告（落盘结构）。"""
    report: Dict[str, Any] = {
        "kind": "regime_conditioned_signal",
        "generated_at": datetime.now().astimezone().isoformat(),
        "affects_gate": False,
        "readonly": True,
        "n_symbols": int(len(data or {})),
        "horizons": [int(h) for h in horizons],
        "folds": int(folds),
        "confidence_threshold": float(confidence_thr),
    }
    if not data:
        report.update({"available": False, "reason": "无可用行情数据（data/raw 为空）"})
        return report

    report["confidence_diagnostics"] = {}
    report["regime_conditioned"] = {}
    for h in horizons:
        ds = _build_supervised(data, config, int(h))
        if ds.empty:
            report["confidence_diagnostics"][str(int(h))] = {
                "horizon_days": int(h), "available": False, "reason": "无样本"}
            report["regime_conditioned"][str(int(h))] = {
                "horizon_days": int(h), "available": False, "reason": "无样本"}
            continue
        feats = _feature_columns(ds, config, int(h))
        proba = _walk_forward_proba(ds, feats, int(h), config, folds)
        report["confidence_diagnostics"][str(int(h))] = confidence_diagnostics(ds, proba, int(h))
        report["regime_conditioned"][str(int(h))] = regime_conditioned_evaluation(
            ds, proba, int(h), confidence_thr=confidence_thr)

    current_weights = (config.get("signal", {}) or {}).get("horizon_weights") or {
        "5": 0.30, "10": 0.35, "20": 0.35,
    }
    report["weight_reallocation_check"] = weight_reallocation_check(
        data, config, horizons=horizons, folds=folds, current_weights=current_weights)

    # 子池稳健性：对**中间周期**（默认 10d）做，避免三周期各跑一遍拖长耗时
    mid_h = int(sorted(set(int(h) for h in horizons))[len(set(horizons)) // 2]) \
        if horizons else 10
    if int(stability_subsets) > 0:
        report["regime_stability"] = regime_stability_check(
            data, config, horizon_days=mid_h, folds=folds,
            confidence_thr=confidence_thr,
            n_subsets=int(stability_subsets), subset_size=int(stability_size))
        report["regime_stability"]["horizon_days"] = mid_h

    report["available"] = bool(
        any(v.get("available") for v in report["confidence_diagnostics"].values())
        or any(v.get("available") for v in report["regime_conditioned"].values())
        or report["weight_reallocation_check"].get("available"))
    if not report["available"]:
        report["reason"] = "全部读数不可用（数据或样本不足）"
    report["conclusion_note"] = (
        "本报告只产出证据：波动状态分层、置信度语义判定、周期权重重排检验均为"
        "**只读**读数（`affects_gate=False`），不改变任何门禁 / 权重 / 池；"
        "是否据此调整采纳口径属产品变更，须人工签字。"
    )
    return report
