"""组合回测闭环基线（S21 / I1：组合层是 G/H 轮之后唯一未验证的层级）。

回答的问题（Issue #54）：
  G/H 轮把「单标的读数可信度」做透了（CPCV 收缩后 IC 仍为正、保形覆盖率
  达标、校准 ECE 0.0904、状态分层 differentiated），但全部评估停留在
  IC / 命中率 / 覆盖率 —— 从未回答「这套信号作为一个 A 股组合，扣完成本
  到底赚不赚钱」。本模块把「信号→仓位→组合净值」补成闭环，并复验 S15
  「IC 与命中率脱节」在组合 PnL 层是否仍成立。

## 执行口径（写死，防自由度回流）

- **MOC 惯例**：权重在 T 日收盘「决定并成交」，持有 (T → T+1) 段，赚取
  T+1 收盘相对 T 收盘的收益。引擎内通过 ``shift(1)`` 强制执行 —— T 日的
  权重绝不影响 T 日的收益（无未来函数，见 tests/test_roadmap_s21.py）。
- **成本**：换手 × 单边成本，逐日从收益中扣减。单边成本取 T11.2 定稿
  三档（``factor_metrics.T112_COST_TIERS``，单一事实源，不得被配置覆盖）。
- **等权归一**：``normalize="equal_active"`` 时，各目标权重除以当日
  「激活标的数」，保证满仓时总权重 = 1；无激活标的当日为空仓。
- **report_only**：本模块是评估工具，不是策略生成器；输出只进报表，
  ``affects_gate`` 恒为 false，不触碰 ``strategy_gate`` 任何字段。

## 不做什么

- 不产出下单指令（延续 Q4 边界：输出止于建议与证据）；
- 不拟合、不调参 —— 给定信号序列只做确定性会计；
- 不猜：日期不可对齐 / 样本不足时如实返回 ``available=False``。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.eval.factor_metrics import T112_COST_TIERS

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252

# 信号 frame 的必需列 / 接受取值
_SIGNAL_REQUIRED_COLS = ("date", "action", "confidence")
_BULLISH_ACTIONS = {"buy", "long", "1", "1.0"}
_PLAN_REQUIRED_COLS = ("date", "target_weight")


def cost_one_side(level: str) -> float:
    """取 T11.2 定稿三档之一的单边成本（佣金 + 滑点）。"""
    for t in T112_COST_TIERS:
        if t["name"] == level:
            return float(t["one_side"])
    raise ValueError(f"未知成本档 {level!r}（可选：{[t['name'] for t in T112_COST_TIERS]}）")


def _norm_date(df: pd.DataFrame, col: str = "date") -> pd.Series:
    return pd.to_datetime(df[col]).dt.normalize()


def build_equal_weight_plan(signal_frames: Dict[str, pd.DataFrame],
                            min_confidence: float = 0.0,
                            ) -> Dict[str, pd.DataFrame]:
    """把信号 frame 转成等权目标权重计划。

    信号 frame 契约：``date`` / ``action``（BUY/SELL/HOLD 或 1/0/-1）/
    ``confidence`` ∈ [0,1]。看多且置信度达阈值 → 目标权重 1.0（激活），
    否则 0.0。逐标的独立产出；**组合内的等权归一在引擎里做**
    （``normalize="equal_active"``），本函数不越权。
    """
    plans: Dict[str, pd.DataFrame] = {}
    for symbol, sig in signal_frames.items():
        missing = [c for c in _SIGNAL_REQUIRED_COLS if c not in sig.columns]
        if missing:
            raise ValueError(f"{symbol} 信号 frame 缺列 {missing}")
        action = sig["action"].astype(str).str.strip().str.lower()
        conf = pd.to_numeric(sig["confidence"], errors="coerce").fillna(0.0)
        active = action.isin(_BULLISH_ACTIONS) & (conf >= float(min_confidence))
        plans[symbol] = pd.DataFrame({
            "date": _norm_date(sig),
            "target_weight": np.where(active, 1.0, 0.0),
        })
    return plans


def run_portfolio_backtest(price_frames: Dict[str, pd.DataFrame],
                           weight_plans: Dict[str, pd.DataFrame],
                           cost_one_side_value: float = 0.0,
                           normalize: Optional[str] = "equal_active",
                           trading_days: int = TRADING_DAYS_PER_YEAR,
                           ) -> Dict[str, Any]:
    """确定性组合回测会计：给定价格与权重计划，产出净值/回撤/换手/夏普。

    Args:
        price_frames: ``{symbol: DataFrame(date, close)}``，按日期升序。
        weight_plans: ``{symbol: DataFrame(date, target_weight)}``，
            ``target_weight`` 为 **T 日收盘决定**的目标权重 ∈ [0, 1]。
        cost_one_side_value: 单边成本（占成交额比例），逐日
            ``换手 × 单边成本`` 扣减。
        normalize: ``"equal_active"`` 等权归一（激活标的等分满仓）；
            ``None`` 直接使用目标权重（总权重可 < 1，不得 > 1）。
        trading_days: 年化用交易日数（缺省 252）。

    Returns:
        含 ``available`` / ``equity``（逐日净值）/ ``metrics``（收益、回撤、
        换手、成本拖累、夏普）/ ``daily``（逐日毛收益/成本/净收益）的 dict；
        日期不可对齐或样本不足时 ``available=False`` + 原因（不猜）。
    """
    if not weight_plans:
        return {"available": False, "reason": "空权重计划", "affects_gate": False}
    unknown = set(weight_plans) - set(price_frames)
    if unknown:
        return {"available": False,
                "reason": f"权重计划包含无价格标的: {sorted(unknown)}",
                "affects_gate": False}

    closes: Dict[str, pd.Series] = {}
    weights: Dict[str, pd.Series] = {}
    for symbol in weight_plans:
        px = price_frames[symbol]
        for col in ("date", "close"):
            if col not in px.columns:
                return {"available": False,
                        "reason": f"{symbol} 价格 frame 缺列 {col!r}",
                        "affects_gate": False}
        plan = weight_plans[symbol]
        missing = [c for c in _PLAN_REQUIRED_COLS if c not in plan.columns]
        if missing:
            return {"available": False,
                    "reason": f"{symbol} 权重计划缺列 {missing}",
                    "affects_gate": False}
        if not plan["target_weight"].between(0.0, 1.0).all():
            return {"available": False,
                    "reason": f"{symbol} 目标权重越界（必须在 [0,1]，不做杠杆）",
                    "affects_gate": False}
        closes[symbol] = pd.Series(
            pd.to_numeric(px["close"], errors="coerce").to_numpy(),
            index=_norm_date(px)).dropna()
        w = pd.Series(pd.to_numeric(plan["target_weight"], errors="coerce").to_numpy(),
                      index=_norm_date(plan)).dropna()
        weights[symbol] = w[~w.index.duplicated(keep="last")]

    dates = sorted(set.intersection(*[set(w.index) for w in weights.values()]))
    if len(dates) < 3:
        return {"available": False,
                "reason": f"共同交易日不足（仅 {len(dates)} 天），拒绝给出读数",
                "affects_gate": False}
    idx = pd.DatetimeIndex(dates)

    W = pd.DataFrame({s: weights[s].reindex(idx).fillna(0.0) for s in weights})
    if normalize == "equal_active":
        active = (W > 0).sum(axis=1)
        scale = pd.Series(np.where(active > 0, 1.0 / active.replace(0, np.nan), 0.0),
                          index=idx)
        W = W.mul(scale, axis=0).fillna(0.0)
    elif normalize is not None:
        return {"available": False, "reason": f"未知归一方式 {normalize!r}",
                "affects_gate": False}
    if (W.sum(axis=1) > 1.0 + 1e-9).any():
        return {"available": False, "reason": "总权重 > 1（不做杠杆），拒绝执行",
                "affects_gate": False}

    px = pd.DataFrame({s: closes[s].reindex(idx) for s in closes})
    nan_symbols = [s for s in closes if px[s].isna().any()]
    if nan_symbols:
        return {"available": False,
                "reason": f"共同交易日内价格缺失: {sorted(nan_symbols)}"
                          "（停牌请在上游显式补齐，引擎不猜）",
                "affects_gate": False}
    # 首行无前一日 → 收益记 0（窗口起点），其余按收盘对收盘
    R = px.pct_change().fillna(0.0)

    # 无未来函数的关键一行：T 日收益只吃 T-1 日收盘决定的权重
    gross = (W.shift(1).fillna(0.0) * R).sum(axis=1)
    # 换手含首日建仓（窗口起点视为空仓）：T 日收盘成交，成本当日计
    turnover = (W - W.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = turnover * float(cost_one_side_value)
    net = gross - cost

    equity = (1.0 + net).cumprod()
    n = int(len(idx))
    total_return = float(equity.iloc[-1] - 1.0)
    ann_return = float((1.0 + total_return) ** (trading_days / n) - 1.0) if n else None
    vol = float(net.std(ddof=0))
    ann_vol = float(vol * np.sqrt(trading_days))
    sharpe = float(net.mean() / vol * np.sqrt(trading_days)) if vol > 1e-12 else None
    peak = equity.cummax()
    max_drawdown = float((equity / peak - 1.0).min())

    daily = pd.DataFrame({
        "date": idx.strftime("%Y-%m-%d"),
        "gross": gross.round(8).to_numpy(),
        "cost": cost.round(8).to_numpy(),
        "net": net.round(8).to_numpy(),
        "equity": equity.round(8).to_numpy(),
        "turnover": turnover.round(8).to_numpy(),
        "n_active": (W > 0).sum(axis=1).to_numpy(),
    })

    return {
        "available": True,
        "affects_gate": False,
        "report_only": True,
        "n_days": n,
        "date_range": [idx[0].strftime("%Y-%m-%d"), idx[-1].strftime("%Y-%m-%d")],
        "n_symbols": len(weights),
        "metrics": {
            "total_return": round(total_return, 8),
            "annualized_return": round(ann_return, 8) if ann_return is not None else None,
            "annualized_vol": round(ann_vol, 8),
            "sharpe": round(sharpe, 8) if sharpe is not None else None,
            "max_drawdown": round(max_drawdown, 8),
            "total_cost_drag": round(float(cost.sum()), 8),
            "avg_daily_turnover": round(float(turnover.mean()), 8),
            "n_active_days": int((W.sum(axis=1) > 0).sum()),
        },
        "daily": daily,
        "convention": {
            "execution": "MOC：T 日收盘决定并成交，持有 T→T+1（shift(1) 强制）",
            "cost": f"换手 × 单边成本 {cost_one_side_value:.5f}，逐日扣减",
            "normalize": normalize or "none",
        },
    }


def _per_symbol_evidence(symbol: str, px: pd.DataFrame,
                         plan: pd.DataFrame) -> Dict[str, Any]:
    """单标的证据：命中率（持仓日方向正确占比）与 IC（权重 vs 次日收益 spearman）。

    两个读数与既有评估口径同源：命中率是门禁现行指标（min_hit_rate 0.52
    只读引用），IC 是 S5 以来的一贯量尺。这里只做**只读复算**，不改任何阈值。
    """
    idx = pd.DatetimeIndex(_norm_date(px))
    close = pd.Series(pd.to_numeric(px["close"], errors="coerce").to_numpy(),
                      index=idx).dropna()
    w = pd.Series(
        pd.to_numeric(plan["target_weight"], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(_norm_date(plan))).dropna()
    w = w[~w.index.duplicated(keep="last")].reindex(close.index).fillna(0.0)
    ret = close.pct_change()

    held = w.shift(1).fillna(0.0) > 0          # T 日持仓 = T-1 日权重
    r = ret
    if held.sum() == 0:
        return {"symbol": symbol, "available": False,
                "reason": "全程空仓，无命中率/IC 可言"}
    hit_rate = float((r[held] > 0).mean())
    ic = float(w.corr(r.shift(-1), method="spearman")) if w.std() > 0 else None
    return {"symbol": symbol, "available": True,
            "n_held_days": int(held.sum()),
            "hit_rate": round(hit_rate, 6),
            "ic": round(ic, 6) if ic is not None and np.isfinite(ic) else None}


def consistency_contrast(price_frames: Dict[str, pd.DataFrame],
                         weight_plans: Dict[str, pd.DataFrame],
                         cost_one_side_value: float,
                         trading_days: int = TRADING_DAYS_PER_YEAR,
                         ) -> Dict[str, Any]:
    """口径一致性对照（T21.2）：IC/命中率读数 vs 组合 PnL 读数是否同向。

    对每个标的独立跑一遍同引擎回测（满仓单标的），得到净收益；再与
    命中率 / IC 做跨标的秩相关与皮尔逊相关。S15 的「IC 与命中率脱节」
    若在 PnL 层仍成立，则相关应显著为正；**相关为负或 |ρ| 极低即为背离**，
    如实入库，不择优、不修参数。
    """
    rows: List[Dict[str, Any]] = []
    for symbol, plan in weight_plans.items():
        ev = _per_symbol_evidence(symbol, price_frames[symbol], plan)
        single = run_portfolio_backtest(
            {symbol: price_frames[symbol]}, {symbol: plan},
            cost_one_side_value=cost_one_side_value, normalize=None,
            trading_days=trading_days)
        pnl = single["metrics"]["total_return"] if single.get("available") else None
        rows.append({
            "symbol": symbol,
            "hit_rate": ev.get("hit_rate"),
            "ic": ev.get("ic"),
            "net_return": round(pnl, 8) if pnl is not None else None,
            "n_held_days": ev.get("n_held_days"),
        })
    ok = [r for r in rows
          if r["hit_rate"] is not None and r["net_return"] is not None]
    if len(ok) < 3:
        return {"available": False,
                "reason": f"可对照标的不足（{len(ok)} < 3），拒绝给相关读数",
                "rows": rows, "affects_gate": False}
    hr = pd.Series([r["hit_rate"] for r in ok], index=[r["symbol"] for r in ok])
    icv = pd.Series([r["ic"] for r in ok if r["ic"] is not None],
                    index=[r["symbol"] for r in ok if r["ic"] is not None])
    pnl = pd.Series([r["net_return"] for r in ok], index=[r["symbol"] for r in ok])
    common = hr.index.intersection(pnl.index)

    divergent = [
        {"symbol": s, "hit_rate": float(hr[s]), "net_return": float(pnl[s]),
         "kind": ("hit_ge_50pct_but_loss" if pnl[s] <= 0
                  else "hit_lt_50pct_but_profit")}
        for s in common
        if (hr[s] >= 0.5) != (pnl[s] > 0)
    ]
    return {
        "available": True,
        "affects_gate": False,
        "n_symbols": int(len(common)),
        "corr_hit_rate_vs_pnl_pearson": (round(float(hr[common].corr(pnl)), 6)
                                         if len(common) >= 3 else None),
        "corr_hit_rate_vs_pnl_spearman": (round(float(hr[common].corr(pnl, method="spearman")), 6)
                                          if len(common) >= 3 else None),
        "corr_ic_vs_pnl_spearman": (round(float(icv.corr(pnl[icv.index], method="spearman")), 6)
                                    if len(icv) >= 3 and pnl[icv.index].std() > 0 else None),
        "divergent_symbols": divergent,
        "verdict_note": ("命中率/IC 与组合 PnL 同向为「一致」；相异标的逐只列出，"
                         "不修参数、不择优 —— 口径取舍属 T21.4 人工检查点"),
        "rows": rows,
    }
