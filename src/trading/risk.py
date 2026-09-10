"""风险管理：仓位计算、止损止盈与风险限额。

风控器根据账户权益与配置的风险参数，把交易信号转化为：
  - 目标仓位（建议下单金额 / 数量）
  - 止损价、止盈价（由价格比例推算）
提供固定分数法与凯利公式（Kelly，以正赔率上界截断）两种仓位模型。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# 凯利公式可能给出极端大仓位，做上界截断防止过度集中
DEFAULT_MAX_KELLY_FRACTION = 0.25
DEFAULT_MAX_POSITION_FRACTION = 0.20  # 单标的占用总资金上限
DEFAULT_BASE_POSITION_PCT = 0.10       # 固定分数法的基础仓位


@dataclass
class RiskBudget:
    """一个交易机会的风控建议。"""

    symbol: str
    action: str
    capital: float
    risk_per_trade: float
    position_amount: float
    position_pct: float
    suggested_qty: int | float | None
    stop_price: float | None
    take_price: float | None
    method: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "capital": round(self.capital, 2),
            "risk_per_trade": round(self.risk_per_trade, 2),
            "position_amount": round(self.position_amount, 2),
            "position_pct": round(self.position_pct, 4),
            "suggested_qty": self.suggested_qty,
            "stop_price": self.stop_price,
            "take_price": self.take_price,
            "method": self.method,
        }


class RiskManager:
    """风控器：基于账户与配置输出仓位与止损止盈。"""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = (config or {}).get("trading", {}).get("risk", {})
        self.capital = float(cfg.get("capital", 1_000_000))
        # 单笔可承受亏损占资金比例
        self.risk_per_trade_pct = float(cfg.get("risk_per_trade_pct", 0.01))
        self.max_position_pct = float(cfg.get("max_position_pct", DEFAULT_MAX_POSITION_FRACTION))
        self.base_position_pct = float(cfg.get("base_position_pct", DEFAULT_BASE_POSITION_PCT))
        self.method = cfg.get("sizing_method", "fixed_fractional")  # fixed_fractional / kelly
        self.max_kelly = float(cfg.get("max_kelly_fraction", DEFAULT_MAX_KELLY_FRACTION))
        # 止损 / 止盈比例（无行情波动信息时按价格比例回退）
        self.stop_loss_pct = float(cfg.get("stop_loss_pct", 0.03))
        self.take_profit_pct = float(cfg.get("take_profit_pct", 0.06))
        signal_cfg = (config or {}).get("trading", {}).get("signal", {})
        self.allow_short = bool(signal_cfg.get("allow_short", True))

    def _sizing_fraction(self, confidence: float, strength: float,
                         proba_win: float | None, price: float) -> float:
        """返回目标仓位占可用资金的比例。

        固定分数法：base * (0.5+0.5*strength) * (0.5+0.5*confidence)。
        凯利法：f=(p*b-q)/b，p 用置信度近似，b 由盈亏比决定，上界截断。
        """
        if self.method == "kelly" and proba_win is not None:
            p = min(max(proba_win, 0.5), 0.95)
            b = self.take_profit_pct / max(self.stop_loss_pct, 1e-6)  # 盈亏比
            q = 1.0 - p
            f = (p * b - q) / b
            return max(0.0, min(f, self.max_kelly))

        # 固定分数法
        f = self.base_position_pct * (0.5 + min(strength, 1.0) * 0.5)
        f = f * (0.5 + min(confidence, 1.0) * 0.5)
        return max(0.0, min(f, self.max_position_pct))

    def budget(self, signal, price: float | None = None,
               proba_win: float | None = None) -> RiskBudget:
        """根据信号与最新价计算风控预算。

        Args:
            signal   : signal.Signal 对象（含 action/confidence/strength/symbol）
            price    : 当前价格（用于算建议数量与止损止盈价）
            proba_win: 胜率估计（凯利法用，默认取信号置信度）
        """
        if signal.action == "HOLD":
            return RiskBudget(
                symbol=signal.symbol, action="HOLD", capital=self.capital,
                risk_per_trade=0.0, position_amount=0.0, position_pct=0.0,
                suggested_qty=None, stop_price=None, take_price=None, method=self.method,
            )

        long_side = signal.action == "BUY"
        conf = signal.confidence
        strength = signal.strength
        p_win = proba_win if proba_win is not None else conf

        frac = self._sizing_fraction(conf, strength, p_win, price or 1.0)
        position_amount = self.capital * frac
        risk_amount = self.capital * self.risk_per_trade_pct

        # 止损止盈价
        stop = take = None
        qty = None
        if price and price > 0:
            if long_side:
                stop = round(price * (1 - self.stop_loss_pct), 4)
                take = round(price * (1 + self.take_profit_pct), 4)
            else:
                stop = round(price * (1 + self.stop_loss_pct), 4)
                take = round(price * (1 - self.take_profit_pct), 4)
            qty = int(position_amount // price) if position_amount else 0
            if qty <= 0 and position_amount > 0:
                qty = 1  # 至少 1 股/手（示意）

        budget = RiskBudget(
            symbol=signal.symbol,
            action=signal.action,
            capital=self.capital,
            risk_per_trade=round(risk_amount, 2),
            position_amount=round(position_amount, 2),
            position_pct=frac,
            suggested_qty=qty,
            stop_price=stop,
            take_price=take,
            method=self.method,
        )
        logger.info(
            "风控 %s %s: 建议金额=%.0f qty=%s 止损=%s 止盈=%s",
            signal.symbol, signal.action, budget.position_amount,
            budget.suggested_qty, budget.stop_price, budget.take_price,
        )
        return budget
