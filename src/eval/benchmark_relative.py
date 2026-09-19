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
- 全部结论带 `verdict` 分级：`no_edge` / `inconclusive` / `positive`；
- **状态分层**（`regime_breakdown`）：三分（bull/range/bear）+ **趋势/盘整二分**
  （`binary`，bull∪bear→trending、range→choppy）—— 二分用于回答三分回答不了的
  "趋势 vs 盘整"（expanding 口径下 bull 可能被吸收）。两者同源，只做展示归并。

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
from src.eval.regime import (REGIME_BINARY_MAP, REGIME_BINARY_ORDER,
                             REGIME_CHOPPY_MEMBERS, REGIME_TRENDING_MEMBERS)

logger = logging.getLogger(__name__)

CURRENT_WEIGHTS = {"5": 0.30, "10": 0.35, "20": 0.35}
REGIME_ORDER = ("bull", "range", "bear")   # 可读化排序（与 regime.py 同集合）
MIN_REGIME_PERIODS = 8                     # 单状态调仓期数下限（不足不发读数）
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
# 状态分层的净超额（"信号是不是只在某种市场状态下才有增量"）
# ----------------------------------------------------------------------
def _date_regime_labels(price_frames: Dict[str, pd.DataFrame],
                        refit_every: int = 20,
                        window: int = 20) -> Dict[str, Any]:
    """由**全池等权市场层价格**拟合出逐交易日状态标签（无前视）。

    观测源与 `main.py::_build_market_series` 同口径：逐标的日收益 → 按日期
    对齐等权平均 → 累乘回价格。状态严格只用当日及之前信息：

      - 首选 `regime.regime_labels`（HMM expanding 口径，无前视）；
      - hmmlearn 缺失 / 拟合退化 → 回落 `regime._rules_states`（同样无前视），
        并在 `mode` 里**如实标注** `rules_fallback`，不冒充 HMM。

    返回 ``{"labels_by_date": {date: 状态|None}, "meta": {...}}``。
    拟合不可用时 `labels_by_date` 为空 dict（调用方如实报不可用，不猜）。
    """
    from src.eval import regime as rg

    frames = []
    for symbol, px in (price_frames or {}).items():
        if px is None or len(px) < 2 or "date" not in px.columns or "close" not in px.columns:
            continue
        d = pd.to_datetime(px["date"], errors="coerce")
        c = pd.to_numeric(px["close"], errors="coerce")
        s = pd.DataFrame({"date": d, "close": c}).dropna().sort_values("date")
        if len(s) < 2:
            continue
        s["ret"] = s["close"].astype(float).pct_change()
        frames.append(s[["date", "ret"]].rename(columns={"ret": symbol}))
    meta: Dict[str, Any] = {"n_symbols": len(frames), "refit_every": int(refit_every),
                            "window": int(window)}
    if not frames:
        meta.update({"available": False, "reason": "no_price_frames", "mode": "none"})
        return {"labels_by_date": {}, "meta": meta}

    mkt = frames[0]
    for f in frames[1:]:
        mkt = mkt.merge(f, on="date", how="outer")
    mkt = mkt.sort_values("date").reset_index(drop=True)
    ret_cols = [c for c in mkt.columns if c != "date"]
    mean_ret = (mkt[ret_cols].mean(axis=1, skipna=True).fillna(0.0)
                .to_numpy(dtype=float))
    close = np.cumprod(1.0 + mean_ret)
    dates = list(mkt["date"])

    obs = rg.build_observations(close, window=int(window))
    if obs.get("reason"):
        meta["observation_reason"] = obs["reason"]
    mode = "hmm_expanding"
    try:
        res = rg.regime_labels(obs["X"], obs["valid"], refit_every=int(refit_every))
    except Exception as e:  # noqa: BLE001
        # fail-soft：**状态标签本身**失败（hmmlearn 未装 / 依赖 ABI / 数值炸）
        # 必须降级到规则口径，而不是把异常抛给调用方 —— 否则一个状态层读数
        # 失败会连带把主读数也标成不可用（"读数本来可用却被记成不可用"）。
        # 不静默降级：异常类型写进 meta，且 mode 如实标 rules_fallback。
        res = {"labels": [], "meta": {"available": False,
                                      "reason": f"{type(e).__name__}: {e}"}}
    labels = list(res.get("labels") or [])
    meta["hmm_meta"] = dict(res.get("meta") or {})
    if not (res.get("meta") or {}).get("available"):
        # 如实降级：规则口径（同样无前视），并标注真实口径
        mode = "rules_fallback"
        labels = list(rg._rules_states(np.diff(np.log(np.maximum(close, 1e-12)),
                                               prepend=0.0), window=int(window)))
        meta["fallback_reason"] = (res.get("meta") or {}).get("reason") or "hmm_unavailable"
    labels_by_date = {d: lab for d, lab in zip(dates, labels)}
    avail = any(v is not None for v in labels_by_date.values())
    meta.update({"available": bool(avail), "mode": mode,
                 "n_days": int(len(labels_by_date)),
                 "n_labeled": int(sum(1 for v in labels_by_date.values() if v)),
                 "lookahead_prefixed": False})
    if not avail:
        meta.setdefault("reason", "no_regime_labels")
        return {"labels_by_date": {}, "meta": meta}
    return {"labels_by_date": labels_by_date, "meta": meta}


def _binary_regime_breakdown(grid: Sequence[pd.Timestamp],
                             labels_by_date: Dict[Any, Optional[str]],
                             signal: np.ndarray, benchmark: np.ndarray,
                             one_side_cost: float,
                             horizon_days: int) -> Dict[str, Any]:
    """在三分读数之外，再给**趋势 / 盘整二分**（同一批逐期净超额，换分组）。

    为什么需要它：expanding 口径下三分可能出现**牛态无有效期数**
    （强趋势样本把牛态吸收掉），导致状态分层只有 range/bear 两态可比，
    "趋势 vs 盘整"这一问**没被回答**。二分把 bull+bear 归并成 trending，
    让该问一定有读数 —— 这是**展示归并**，不新增拟合口径、不改判据。

    判据、无前视、最小期数纪律与 `_regime_breakdown` 完全一致。
    """
    n = min(len(signal), len(benchmark), len(grid))
    norm_labels = {pd.Timestamp(k): v for k, v in (labels_by_date or {}).items()}
    groups: Dict[str, List[float]] = {name: [] for name in REGIME_BINARY_ORDER}
    unlabeled = 0
    for i in range(n):
        raw = norm_labels.get(pd.Timestamp(grid[i]))
        lab = REGIME_BINARY_MAP.get(raw or "")
        if lab not in groups:
            unlabeled += 1
            continue
        d = float(signal[i]) - float(benchmark[i]) - 2.0 * float(one_side_cost)
        if np.isfinite(d):
            groups[lab].append(d)

    per: Dict[str, Any] = {}
    usable: List[str] = []
    for name in REGIME_BINARY_ORDER:
        arr = np.asarray(groups[name], dtype=float)
        st: Dict[str, Any] = {"name": name, "n_periods": int(arr.size)}
        if arr.size < MIN_REGIME_PERIODS:
            st.update({"available": False,
                       "reason": f"该状态调仓期数不足（{arr.size} < {MIN_REGIME_PERIODS}）"})
            per[name] = st
            continue
        sd = float(arr.std(ddof=1))
        t = float(arr.mean() / sd * np.sqrt(arr.size)) if sd > 1e-12 else None
        st.update({
            "available": True,
            "excess_mean_per_period": _round(float(arr.mean()), 8),
            "excess_t_stat": _round(t, 4) if t is not None else None,
            "positive": bool(arr.mean() > 0),
            "share": _round(float(arr.size) / max(1, n), 6),
        })
        per[name] = st
        usable.append(name)

    out: Dict[str, Any] = {
        "available": bool(len(usable) >= 2),
        "n_periods_total": int(n),
        "n_periods_unlabeled": int(unlabeled),
        "by_regime": per,
        "min_periods": int(MIN_REGIME_PERIODS),
        "regime_order": list(REGIME_BINARY_ORDER),
        "mapping": {k: list(v) for k, v in
                    (("trending", list(REGIME_TRENDING_MEMBERS)),
                     ("choppy", list(REGIME_CHOPPY_MEMBERS)))},
        "kind": "binary_trend_choppy",
    }
    if len(usable) >= 2:
        means = {k: per[k]["excess_mean_per_period"] for k in usable}
        best = max(means, key=lambda k: means[k])
        worst = min(means, key=lambda k: means[k])
        out["spread"] = {
            "best": best, "best_excess": means[best],
            "worst": worst, "worst_excess": means[worst],
            "best_minus_worst": _round(float(means[best] - means[worst]), 8),
        }
        sig_pos = [k for k in usable
                   if (per[k].get("excess_t_stat") or 0) >= 2.0
                   and (per[k].get("excess_mean_per_period") or 0) > 0]
        out["regimes_with_positive_edge"] = sorted(sig_pos)
        out["conditional_edge_hint"] = bool(sig_pos)
        out["conclusion"] = (
            "conditional_hint" if sig_pos else "no_conditional_edge")
    else:
        out["reason"] = "可比状态不足（可用状态 < 2），拒绝给出趋势/盘整分层结论"
        out["conditional_edge_hint"] = False
        out["conclusion"] = "unavailable"
    return out


def _regime_breakdown(grid: Sequence[pd.Timestamp],
                      labels_by_date: Dict[Any, Optional[str]],
                      signal: np.ndarray, benchmark: np.ndarray,
                      one_side_cost: float, horizon_days: int) -> Dict[str, Any]:
    """把**逐期净超额**按调仓日的市场状态分层，逐状态给读数。

    判据与全局完全同源：逐期 ``(signal − benchmark) − 2×单边成本``。
    调仓日状态取**当日**标签（无前视 —— 标签只由 ≤ 当日的观测拟合）。

    分隔组按 `REGIME_ORDER` 排序；单状态可比期数 < `MIN_REGIME_PERIODS`
    如实标 `available=False` + reason，**不外推该状态结论**。

    关键读数 `spread`：状态间净超额之差（最大 − 最小），用于回答
    「增量是不是只集中在某个状态」——若极差落在噪声内（无状态显著
    高于 0、且状态间 t 不显著），不得声称"条件 edge"。
    """
    n = min(len(signal), len(benchmark), len(grid))
    norm_labels = {pd.Timestamp(k): v for k, v in (labels_by_date or {}).items()}
    groups: Dict[str, List[float]] = {name: [] for name in REGIME_ORDER}
    unlabeled = 0
    for i in range(n):
        lab = norm_labels.get(pd.Timestamp(grid[i]))
        if lab not in groups:
            unlabeled += 1
            continue
        d = float(signal[i]) - float(benchmark[i]) - 2.0 * float(one_side_cost)
        if np.isfinite(d):
            groups[lab].append(d)

    per: Dict[str, Any] = {}
    usable: List[str] = []
    for name in REGIME_ORDER:
        arr = np.asarray(groups[name], dtype=float)
        st: Dict[str, Any] = {"name": name, "n_periods": int(arr.size)}
        if arr.size < MIN_REGIME_PERIODS:
            st.update({"available": False,
                       "reason": f"该状态调仓期数不足（{arr.size} < {MIN_REGIME_PERIODS}）"})
            per[name] = st
            continue
        sd = float(arr.std(ddof=1))
        t = float(arr.mean() / sd * np.sqrt(arr.size)) if sd > 1e-12 else None
        st.update({
            "available": True,
            "excess_mean_per_period": _round(float(arr.mean()), 8),
            "excess_t_stat": _round(t, 4) if t is not None else None,
            "positive": bool(arr.mean() > 0),
            "share": _round(float(arr.size) / max(1, n), 6),
        })
        per[name] = st
        usable.append(name)

    out: Dict[str, Any] = {
        "available": bool(len(usable) >= 2),
        "n_periods_total": int(n),
        "n_periods_unlabeled": int(unlabeled),
        "by_regime": per,
        "min_periods": int(MIN_REGIME_PERIODS),
    }
    # 与调用方同口径的状态展示顺序（只对出现在 per 里的状态排序）
    out["regime_order"] = ([nm for nm in REGIME_ORDER if nm in per]
                          + [nm for nm in per if nm not in REGIME_ORDER])
    if len(usable) >= 2:
        means = {k: per[k]["excess_mean_per_period"] for k in usable}
        best = max(means, key=lambda k: means[k])
        worst = min(means, key=lambda k: means[k])
        out["spread"] = {
            "best": best, "best_excess": means[best],
            "worst": worst, "worst_excess": means[worst],
            "best_minus_worst": _round(float(means[best] - means[worst]), 8),
        }
        # 只在某状态净超额显著为正（t≥2）时才允许说"条件 edge"的**迹象**
        sig_pos = [k for k in usable
                   if (per[k].get("excess_t_stat") or 0) >= 2.0
                   and (per[k].get("excess_mean_per_period") or 0) > 0]
        out["regimes_with_positive_edge"] = sorted(sig_pos)
        out["conditional_edge_hint"] = bool(sig_pos)
        out["conclusion"] = (
            "conditional_hint" if sig_pos else "no_conditional_edge")
    else:
        out["reason"] = "可比状态不足（可用状态 < 2），拒绝给出状态分层结论"
        out["conditional_edge_hint"] = False
        out["conclusion"] = "unavailable"
    return out


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
                 seed: int = 7,
                 regime_breakdown: bool = True,
                 regime_refit_every: int = 20,
                 regime_window: int = 20) -> Dict[str, Any]:
    """基准相对决策增量评估（只读）。

    Args:
        data: ``{symbol: DataFrame(date, close)}`` 价格表。
        probabilities: ``{h: DataFrame(date, _symbol, _p)}`` 逐周期**样本外**概率。
        holding_horizon: 非重叠持有期（调仓间隔），默认 5 日。
        confidence_thr: 选中阈值，作用在**综合分**上（默认 0.5 = 中性）。
        cost_level: T11.2 定稿三档之一（conservative / base / aggressive）。
        n_random_controls: 随机子集对照次数。
        regime_breakdown: 是否附**状态分层净超额**（默认开；回答"增量是否只在
            某种市场状态下存在"，与全局判据同源，只读数不生效）。
        regime_refit_every / regime_window: 状态拟合口径（透传 `regime` 模块）。

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
    primary_signal: Optional[np.ndarray] = None
    selmap_current: Dict[pd.Timestamp, List[str]] = {}
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
        if name == "current":
            primary_signal = sig
            selmap_current = selmap
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

    # ---- 状态分层的净超额（"增量是不是只在某个状态"） ----
    out["regime_breakdown_enabled"] = bool(regime_breakdown)
    if regime_breakdown:
        rb: Dict[str, Any] = {"available": False, "reason": "未计算"}
        try:
            lab = _date_regime_labels(price_frames,
                                      refit_every=int(regime_refit_every),
                                      window=int(regime_window))
            if not primary.get("available") or primary_signal is None:
                rb = {"available": False, "reason": "主读数不可用，状态分层跳过",
                      "labels": lab.get("meta", {})}
            elif not (lab.get("meta") or {}).get("available"):
                rb = {"available": False,
                      "reason": "状态标签不可用（缺 hmmlearn 且规则口径也退化）",
                      "labels": lab.get("meta", {})}
            else:
                rb = _regime_breakdown(grid, lab.get("labels_by_date") or {},
                                       primary_signal, benchmark, one_side,
                                       holding_horizon)
                rb["labels"] = lab.get("meta", {})
                # 趋势 / 盘整二分：三分出现"牛态无期数"时该问仍有读数
                try:
                    rb["binary"] = _binary_regime_breakdown(
                        grid, lab.get("labels_by_date") or {},
                        primary_signal, benchmark, one_side, holding_horizon)
                except Exception as e:  # noqa: BLE001 - 二分失败不拖垮三分读数
                    rb["binary"] = {"available": False,
                                    "reason": f"趋势/盘整分层失败: {e}"}
        except Exception as e:  # noqa: BLE001 - 状态分层失败不拖垮主读数
            logger.warning("状态分层读数失败，主读数不受影响: %s", e)
            rb = {"available": False, "reason": f"状态分层计算失败: {e}"}
        out["regime_breakdown"] = rb
    else:
        out["regime_breakdown"] = {"available": False, "reason": "已关闭（regime_breakdown=False）"}

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
