# backtest/run_minimal.py
"""
最小回测入口。

职责：
- 接受价格序列和信号序列
- 按信号生成仓位
- 计算简单收益路径
- 输出基础绩效指标

使用方式示例:
    python backtest/run_minimal.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np


def signals_to_positions(
    signals: Sequence[int],
    initial_position: int = 0,
) -> List[int]:
    """
    将信号序列转换为仓位序列。

    约定:
    - 信号为 1 时买入/持有
    - 信号为 0 时卖出/不持有
    - 仓位在信号变化时更新
    """
    positions: List[int] = []
    position = initial_position
    for s in signals:
        if int(s) == 1:
            position = 1
        else:
            position = 0
        positions.append(position)
    return positions


def simulate_returns(
    prices: Sequence[float],
    positions: Sequence[int],
) -> np.ndarray:
    """
    根据价格和仓位模拟资产收益路径。

    返回:
        每期策略收益
    """
    prices = np.asarray(prices, dtype=float)
    positions = np.asarray(positions, dtype=float)

    if len(prices) < 2:
        return np.array([0.0])

    rets = np.diff(prices) / prices[:-1]
    strategy_rets = rets * positions[:-1]
    return strategy_rets


def compute_metrics(
    prices: Sequence[float],
    positions: Sequence[int],
) -> Dict[str, Union[float, int]]:
    """
    计算最小回测指标集。

    指标包括:
    - 累计收益
    - 年化收益(假设252期)
    - 最大回撤
    - 盈利期比例
    - 持仓期数
    """
    prices = np.asarray(prices, dtype=float)
    positions = np.asarray(positions, dtype=float)

    if len(prices) < 2:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "holding_periods": 0,
            "total_periods": int(len(prices)),
        }

    rets = np.diff(prices) / prices[:-1]
    strategy_rets = rets * positions[:-1]

    total_return = float(np.prod(1.0 + strategy_rets) - 1.0)
    n = len(strategy_rets)
    annualized_return = float((1.0 + total_return) ** (252.0 / max(n, 1)) - 1.0)

    equity = np.cumprod(1.0 + strategy_rets)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    max_drawdown = float(np.min(drawdown))

    win_rate = float(np.mean(strategy_rets > 0)) if n > 0 else 0.0
    holding_periods = int(np.sum(positions[:-1] > 0))

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "holding_periods": holding_periods,
        "total_periods": int(len(prices)),
    }


def load_simple_case() -> Dict[str, List[float]]:
    """
    加载一个简单的示例案例，用于验证回测流水线。

    返回:
        prices 和 signals
    """
    return {
        "prices": [100.0, 101.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0],
        "signals": [0, 1, 1, 0, 1, 1, 0, 1],
    }


def run(case: Dict[str, List[float]]) -> Dict[str, Union[float, int]]:
    """
    运行一次最小回测。
    """
    prices = case["prices"]
    signals = case["signals"]
    positions = signals_to_positions(signals)
    metrics = compute_metrics(prices, positions)
    metrics["positions"] = positions
    return metrics


def main() -> None:
    case = load_simple_case()
    result = run(case)

    out = {
        "case": case,
        "result": result,
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
