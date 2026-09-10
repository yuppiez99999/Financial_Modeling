"""轻量回测：用历史行情对 TrendCast 信号做假设成交统计。

因系统按需拉取行情并生成"下一周期方向"预测，回测采用事件驱动简化：
给定一组历史 close 价格序列与（模拟或真实）信号序列，在信号日按方向建仓、
在 horizon 日后平仓，统计 胜率 / 平均收益 / 总收益 / 最大回撤。

本模块重在"快速验证信号质量"，非高保真撮合引擎。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    n_trades: int
    win_rate: float
    avg_return: float
    total_return: float
    max_drawdown: float
    profit_factor: float
    equity_curve: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_trades": self.n_trades,
            "win_rate": round(self.win_rate, 4),
            "avg_return": round(self.avg_return, 4),
            "total_return": round(self.total_return, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "profit_factor": round(self.profit_factor, 4),
        }


class Backtester:
    """基于信号列表的假设成交回测器。"""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = (config or {}).get("trading", {}).get("backtest", {})
        # 每笔默认手续费（比例）
        self.fee = float(cfg.get("fee", 0.001))
        self.initial_capital = float(cfg.get("initial_capital", 1_000_000))

    def run(self, closes: Sequence[float], signals: Sequence[dict[str, Any]]) -> BacktestResult:
        """执行回测。

        Args:
            closes  : 按时间顺序的收盘价序列
            signals : 与 closes 对齐的信号字典序列（每项含 action，可选 hold 周期）
        """
        prices = np.asarray(closes, dtype=float)
        n = len(prices)
        if n < 2 or len(signals) != n:
            return BacktestResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, [self.initial_capital])

        equity = float(self.initial_capital)
        curve = [equity]
        wins = losses = 0
        gross_win = gross_loss = 0.0
        returns: list[float] = []

        i = 0
        while i < n:
            sig = signals[i]
            action = sig.get("action", "HOLD")
            if action in ("BUY", "SELL"):
                # 持有至 horizon 期（默认下一根K线）
                horizon = int(sig.get("horizon", 1))
                j = min(i + horizon, n - 1)
                if j <= i:
                    i += 1
                    continue
                entry = prices[i]
                exit_ = prices[j]
                direction = 1 if action == "BUY" else -1
                ret = direction * (exit_ / entry - 1.0) - self.fee * 2
                equity *= (1 + ret)
                curve.append(equity)
                returns.append(ret)
                if ret > 0:
                    wins += 1
                    gross_win += ret
                else:
                    losses += 1
                    gross_loss += -ret
                i = j + 1
            else:
                i += 1
                if i < n:
                    curve.append(equity)

        n_trades = wins + losses
        win_rate = (wins / n_trades) if n_trades else 0.0
        avg_return = float(np.mean(returns)) if returns else 0.0
        total_return = equity / self.initial_capital - 1.0
        pf = (gross_win / gross_loss) if gross_loss > 0 else (gross_win / (1e-6) if gross_win > 0 else 0.0)

        # 最大回撤
        peak = -1e18
        mdd = 0.0
        for e in curve:
            peak = max(peak, e)
            mdd = max(mdd, (peak - e) / peak if peak > 0 else 0.0)

        return BacktestResult(
            n_trades=n_trades, win_rate=win_rate, avg_return=avg_return,
            total_return=total_return, max_drawdown=mdd, profit_factor=pf,
            equity_curve=[round(c, 2) for c in curve],
        )
