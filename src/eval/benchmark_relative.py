"""基准相对、含成本的组合层判据（Issue #55 —— 回答"到底有没有决策增量"）。

## 问题从哪来

`decision-source-contract` / `model-optimization-findings` / `regime-conditioned-signal`
三轮把候选方向逐个排查完，剩下的决策问题是「要不要按证据重排周期权重」。
但在回答它之前，有一个**更靠前**的问题一直没被回答：

> 这套信号作为组合，相对**什么都不做**（等权持有全池），到底有没有增量？

前几轮的读数（IC / 命中率 / 平均已实现收益 / 波动分层）全部是**绝对读数**，
且都在「信号自身样本」内比较。而 26 标的池在 2021-05~2026-09 的等权 buy&hold
年化约 +14.9% —— 也就是说，**任何"平均已实现收益为正"的读数都可能只是市场beta**，
不足以说明信号有决策价值。

## 本模块的判据（写死，防自由度回流）

- **相对基准**：同一批调仓日、同一持有期，**全池等权**为基准；信号组合必须
  跑赢它才算有增量。绝对收益为正**不算**通过。
- **含成本**：每次调仓按 T11.2 定稿三档扣 `2 × 单边成本`（一进一出）。
- **不重叠**：按持有期在日历上取非重叠调仓日（`h` 日一格），避免"重叠样本
  假装独立"把 t 值撑大。
- **无前视**：T 日决定并用 T 日收盘价建仓，赚 T → T+h 的收益；只用 `[.., T]`
  的信息。所有信号来自 walk-forward 样本外概率。
- **随机子集对照**：再抽同样大小的**随机**标的子集当对照。若信号子集的表现
  落在随机子集的分布里，说明"选中这些标的"没有信息含量。

## 输出纪律

- 只报数，`affects_gate=False`，**不**据自动改权重 / 门槛 / 池；
- 样本不足 / 日期不可对齐 → `available=False` + reason，不外推；
- 全部结论带 `verdict` 分级：`no_edge` / `inconclusive` / `positive`。

## 与已有模块的关系

`portfolio_backtest`（S21/I1）回答「组合回测怎么算」；
`regime_conditioned_signal` 回答「置信度是什么语义」。本模块回答
「**相对基准有没有增量**」—— 这是前两者都没有覆盖的那一问。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.eval.portfolio_backtest import cost_one_side

logger = logging.getLogger(__name__)

CURRENT_WEIGHTS = {"5": 0.30, "10": 0.35, "20": 0.35}
TRADING_DAYS_PER_YEAR = 252


def _finite(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _round(v: Any, nd: int = 6) -> Optional[float]:
    f = _finite(v)
    return None if f is None else round(f, nd)


# ----------------------------------------------------------------------
# 组合层：非重叠持有期收益
# ----------------------------------------------------------------------
def _horizon_returns(price_frames: Dict[str, pd.DataFrame],
                     grid: Sequence[pd.Timestamp],
                     selections: Dict[pd.Timestamp, List[str]],
                     horizon_days: int) -> np.ndarray:
    """给定调仓日与逐日选中标的，返回逐期等权组合的原始持有期收益。

    每期收益 = 选中标的「T+h 收盘 / T 收盘 − 1」的等权平均；未选中记 0（空仓）。
    日期或未来价格不可对齐的标的**跳过**（不猜、不填）。
    """
    closes: Dict[str, pd.Series] = {}
    for symbol, px in price_frames.items():
        if "date" not in px.columns or "close" not in px.columns:
            continue
        s = pd.Series(
            pd.to_numeric(px["close"], errors="coerce").to_numpy(),
            index=pd.to_datetime(px["date"]).dt.normalize(),
        )
        closes[symbol] = s[~s.index.duplicated(keep="last")].sort_index()

    out: List[float] = []
    for t in grid:
        sel = selections.get(t) or []
        vals: List[float] = []
        for s in sel:
            c = closes.get(s)
            if c is None or len(c) < 2:
                continue
            try:
                i = c.index.get_loc(t)
            except KeyError:
                continue
            if isinstance(i, slice) or i + horizon_days >= len(c):
                continue
            p0, p1 = c.iloc[i], c.iloc[i + horizon_days]
            if p0 and np.isfinite(p0) and np.isfinite(p1):
                vals.append(float(p1 / p0 - 1.0))
        out.append(float(np.mean(vals)) if vals else 0.0)
    return np.asarray(out, dtype=float)


def _period_stats(returns: np.ndarray, horizon_days: int) -> Dict[str, Any]:
    """逐期收益 → 汇总读数（不扣成本；成本由调用方显式扣）。"""
    r = np.asarray([x for x in returns if np.isfinite(x)], dtype=float)
    n = int(len(r))
    if n < 8:
        return {"available": False, "reason": f"调仓期数不足（{n} < 8），拒绝给出读数",
                "n_periods": n}
    eq = float(np.prod(1.0 + r))
    sd = float(r.std(ddof=0))
    return {
        "available": True,
        "reason": "",
        "n_periods": n,
        "mean_per_period": _round(float(r.mean()), 8),
        "period_hit_rate": _round(float((r > 0).mean()), 6),
        "total_return": _round(eq - 1.0, 8),
        "annualized_return": _round(eq ** (TRADING_DAYS_PER_YEAR / (horizon_days * n)) - 1.0, 8),
        "annualized_vol": _round(sd * np.sqrt(TRADING_DAYS_PER_YEAR / horizon_days), 8),
        "sharpe": (_round(float(r.mean() / sd * np.sqrt(TRADING_DAYS_PER_YEAR / horizon_days)), 6)
                   if sd > 1e-12 else None),
    }


def _excess_stats(signal: np.ndarray, benchmark: np.ndarray,
                  one_side_cost: float, horizon_days: int) -> Dict[str, Any]:
    """信号相对基准的**净超额**：逐期 (signal − benchmark) − 2×单边成本。

    成本只在信号臂计（基准=全池等权 buy&hold，不做调仓，无换手成本）。
    这样超额的符号就是「信号是否值得做」的直接答案。
    """
    n = min(len(signal), len(benchmark))
    if n < 8:
        return {"available": False, "reason": f"可比调仓期数不足（{n} < 8）"}
    d = signal[:n] - benchmark[:n] - 2.0 * float(one_side_cost)
    sd = float(d.std(ddof=1))
    t = float(d.mean() / sd * np.sqrt(n)) if sd > 1e-12 else None
    return {
        "available": True,
        "reason": "",
        "n_periods": int(n),
        "excess_mean_per_period": _round(float(d.mean()), 8),
        "excess_t_stat": _round(t, 4) if t is not None else None,
        "excess_annualized": _round(
            (1.0 + float(d.mean())) ** (TRADING_DAYS_PER_YEAR / horizon_days) - 1.0, 8),
        "one_side_cost": float(one_side_cost),
    }


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------
def build_report(data: Dict[str, pd.DataFrame],
                 probabilities: Dict[int, pd.DataFrame],
                 config: Dict[str, Any],
                 horizons: Sequence[int] = (5, 10, 20),
                 current_weights: Optional[Dict[str, float]] = None,
                 holding_horizon: int = 5,
                 confidence_thr: float = 0.5,
                 cost_level: str = "base",
                 n_random_controls: int = 40,
                 seed: int = 7) -> Dict[str, Any]:
    """基准相对决策增量评估（只读）。

    Args:
        data: ``{symbol: DataFrame(date, close)}`` 价格表。
        probabilities: ``{h: DataFrame(date, _symbol, _p)}`` 逐周期**样本外**概率。
        holding_horizon: 非重叠持有期（调仓间隔），默认 5 日。
        confidence_thr: 选中阈值，作用在**综合分**上（默认 0.5 = 中性）。
        cost_level: T11.2 定稿三档之一（conservative / base / aggressive）。
        n_random_controls: 随机子集对照次数。

    Returns:
        dict，含 `benchmark_relative`（信号 vs 基准）、`horizon_candidates`
        （候选周期权重在同判据下的读数）、`random_subset_control`、`verdict`。
    """
    out: Dict[str, Any] = {
        "kind": "benchmark_relative",
        "available": False,
        "reason": "",
        "affects_gate": False,
        "readonly": True,
        "holding_horizon": int(holding_horizon),
        "confidence_threshold": float(confidence_thr),
        "cost_level": cost_level,
        "current_weights": dict(current_weights or CURRENT_WEIGHTS),
    }
    symbols = sorted((data or {}).keys())
    if len(symbols) < 2:
        out["reason"] = f"可用标的不足（{len(symbols)} < 2）"
        return out
    missing = [h for h in horizons if h not in probabilities or probabilities[h].empty]
    if missing:
        out["reason"] = f"缺少周期样本外概率: {sorted(missing)}"
        return out

    try:
        one_side = cost_one_side(cost_level)
    except ValueError as e:
        out["reason"] = str(e)
        return out

    # 对齐多周期综合分
    merged: Optional[pd.DataFrame] = None
    for h in horizons:
        t = (probabilities[h][["date", "_symbol", "_p"]]
             .rename(columns={"_p": f"p{int(h)}"}))
        merged = t if merged is None else merged.merge(t, on=["date", "_symbol"], how="inner")
    if merged is None or merged.empty:
        out["reason"] = "多周期样本外概率无法对齐"
        return out
    merged["date"] = pd.to_datetime(merged["date"]).dt.normalize()
    merged = (merged.sort_values(["date", "_symbol"], kind="mergesort")
              .reset_index(drop=True))

    all_dates = sorted(merged["date"].unique())
    grid = all_dates[:: int(holding_horizon)]
    if len(grid) < 8:
        out["reason"] = f"非重叠调仓期数不足（{len(grid)} < 8），拒绝给出读数"
        return out

    price_frames = {s: data[s] for s in symbols if s in data}
    benchmark = _horizon_returns(price_frames, grid,
                                 {t: symbols for t in grid}, holding_horizon)
    out["benchmark"] = _period_stats(benchmark, holding_horizon)
    out["benchmark"]["note"] = f"全池等权 buy&hold（{len(symbols)} 标的，无换手成本）"

    # ---- 候选周期权重，在同一判据下比 ----
    cw = {str(k): float(v) for k, v in (current_weights or CURRENT_WEIGHTS).items()}
    hs = [int(h) for h in horizons]
    evidence = {str(h): abs(float(cw.get(str(h), 0.0))) for h in hs}
    tot = sum(evidence.values()) or 1.0
    evidence = {k: v / tot for k, v in evidence.items()}
    cands: Dict[str, Dict[str, float]] = {
        "current": {str(h): cw.get(str(h), 0.0) for h in hs},
        "equal": {str(h): 1.0 / len(hs) for h in hs},
    }
    for h in hs:
        cands[f"only_{h}d"] = {str(x): (1.0 if x == h else 0.0) for x in hs}

    def _comp(weights: Dict[str, float]) -> np.ndarray:
        w = np.array([float(weights.get(str(h), 0.0)) for h in hs], dtype=float)
        s = w.sum()
        w = w / s if s > 0 else w
        return merged[[f"p{h}" for h in hs]].to_numpy(dtype=float) @ w

    cand_out: Dict[str, Any] = {}
    for name, w in cands.items():
        comp = _comp(w)
        selmap: Dict[pd.Timestamp, List[str]] = {}
        for t in grid:
            m = (merged["date"] == t).to_numpy()
            selmap[t] = merged.loc[m & (comp >= float(confidence_thr)), "_symbol"].tolist()
        sig = _horizon_returns(price_frames, grid, selmap, holding_horizon)
        st = _period_stats(sig, holding_horizon)
        st["weights"] = {str(h): _round(w.get(str(h), 0.0), 6) for h in hs}
        st["avg_n_selected"] = _round(
            float(np.mean([len(v) for v in selmap.values()])), 3)
        st["vs_benchmark"] = _excess_stats(sig, benchmark, one_side, holding_horizon)
        cand_out[name] = st
    out["horizon_candidates"] = cand_out

    # ---- 主读数：现行权重 ----
    primary = cand_out.get("current", {})
    out["benchmark_relative"] = {
        "signal": primary.get("available", False),
        "signal_mean_per_period": primary.get("mean_per_period"),
        "benchmark_mean_per_period": out["benchmark"].get("mean_per_period")
        if out["benchmark"].get("available") else None,
        "vs_benchmark": primary.get("vs_benchmark", {"available": False}),
    }

    # ---- 随机子集对照 ----
    rng = np.random.default_rng(int(seed))
    comp_cur = _comp(cands["current"])
    n_sel = int(round(float(np.mean(
        [int(((merged["date"] == t).to_numpy()
              & (comp_cur >= float(confidence_thr))).sum()) for t in grid]))))
    n_sel = max(1, min(n_sel, len(symbols)))
    draws: List[float] = []
    for _ in range(int(n_random_controls)):
        rm = {t: rng.choice(symbols, size=n_sel, replace=False).tolist() for t in grid}
        r = _horizon_returns(price_frames, grid, rm, holding_horizon)
        b = _excess_stats(r, benchmark, one_side, holding_horizon)
        if b.get("available"):
            draws.append(float(b["excess_mean_per_period"]))
    if len(draws) >= 8:
        dr = np.asarray(draws, dtype=float)
        obs = (primary.get("vs_benchmark") or {}).get("excess_mean_per_period")
        out["random_subset_control"] = {
            "available": True,
            "n_draws": int(len(dr)),
            "subset_size": int(n_sel),
            "random_excess_mean": _round(float(dr.mean()), 8),
            "random_excess_sd": _round(float(dr.std(ddof=1)), 8),
            "signal_excess_mean": obs,
            "fraction_random_ge_signal": (_round(float((dr >= obs).mean()), 4)
                                          if obs is not None else None),
            "note": "信号子集落在随机子集分布内 ⇒ 选中这些标的没有信息含量",
        }
    else:
        out["random_subset_control"] = {
            "available": False,
            "reason": f"可用随机对照次数不足（{len(draws)} < 8）",
        }

    out["available"] = bool(primary.get("available"))
    out["verdict"] = _verdict(out)
    return out


def _verdict(report: Dict[str, Any]) -> Dict[str, Any]:
    """结论分级：只在**净超额显著为正且优于随机子集**时才给 positive。

    纪律：绝对收益为正、IC 为正、命中率 >0.5 都不是通过条件 ——
    唯一通过条件是「相对全池等权的净超额为正，且随机子集做不到」。
    """
    vs = (report.get("benchmark_relative") or {}).get("vs_benchmark") or {}
    ctrl = report.get("random_subset_control") or {}
    if not vs.get("available"):
        return {"level": "unavailable", "reason": "净超额读数不可用"}
    t = vs.get("excess_t_stat")
    obs = vs.get("excess_mean_per_period")
    frac = ctrl.get("fraction_random_ge_signal") if ctrl.get("available") else None
    if obs is None:
        return {"level": "unavailable", "reason": "净超额均值为 None"}
    if obs <= 0:
        return {"level": "no_edge",
                "reason": "净超额（相对全池等权、扣成本后）非正 ⇒ 不如直接等权持有",
                "excess_mean_per_period": obs}
    if t is not None and t >= 2.0 and frac is not None and frac <= 0.10:
        return {"level": "positive",
                "reason": "净超额显著为正且优于随机子集",
                "excess_t_stat": t, "fraction_random_ge_signal": frac}
    return {"level": "inconclusive",
            "reason": ("净超额为正但证据不足（t < 2 或随机子集也能做到）"
                       "⇒ 不足以支撑口径变更"),
            "excess_t_stat": t, "fraction_random_ge_signal": frac}
