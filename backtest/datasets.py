# backtest/datasets.py
"""
回测数据结构定义。

目的：
- 明确回测层需要什么样的数据
- 统一价格、信号、成交假设、频率的接口
- 为后续接入真实数据和预测模型提供标准输入格式
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence


@dataclass
class Bar:
    """
    单根行情条目。
    """
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Signal:
    """
    单期信号条目。
    """
    timestamp: datetime
    symbol: str
    signal: int
    # 1: 做多/持有
    # 0: 空仓/不持有
    # 扩展时可增加 -1 表示做空/下跌方向等


@dataclass
class TradeAssumption:
    """
    成交与成本假设。
    """
    commission_rate: float = 0.0003
    slippage_bps: float = 0.0
    allow_short: bool = False


@dataclass
class BacktestDataBundle:
    """
    回测所需的最小数据包。

    包含:
    - 价格条目
    - 信号条目
    - 频率说明
    - 成交假设
    """
    bars: List[Bar]
    signals: List[Signal]
    frequency: str
    trade_assumption: TradeAssumption
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


def make_simple_bundle() -> BacktestDataBundle:
    """
    构造一个简单的示例数据集，用于验证回测流水线。

    返回:
        简单回测数据包
    """
    bars = [
        Bar(datetime(2026, 1, 1), 100.0, 100.5, 99.8, 100.0, 1000.0),
        Bar(datetime(2026, 1, 2), 100.0, 101.5, 100.0, 101.0, 1100.0),
        Bar(datetime(2026, 1, 3), 101.0, 102.5, 100.5, 102.0, 1200.0),
        Bar(datetime(2026, 1, 4), 102.0, 102.0, 100.0, 101.0, 1150.0),
        Bar(datetime(2026, 1, 5), 101.0, 103.5, 101.0, 103.0, 1300.0),
    ]

    signals = [
        Signal(datetime(2026, 1, 1), "TEST", 0),
        Signal(datetime(2026, 1, 2), "TEST", 1),
        Signal(datetime(2026, 1, 3), "TEST", 1),
        Signal(datetime(2026, 1, 4), "TEST", 0),
        Signal(datetime(2026, 1, 5), "TEST", 1),
    ]

    return BacktestDataBundle(
        bars=bars,
        signals=signals,
        frequency="1d",
        trade_assumption=TradeAssumption(),
        start_date=bars[0].timestamp,
        end_date=bars[-1].timestamp,
    )
