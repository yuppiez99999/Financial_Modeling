# backtest/run_real.py
"""基于真实数据的回测入口。"""
from __future__ import annotations

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.data_provider import DataProvider
from backtest.run_minimal import compute_metrics, signals_to_positions


def load_data(symbol: str = "sh000300", recent_n: int = 252):
    dp = DataProvider()
    df = dp.get_index_daily(symbol)
    if recent_n and recent_n < len(df):
        df = df.tail(recent_n).reset_index(drop=True)
    return df


def make_simple_signal(df):
    """简单信号：收盘价 > 20日均线则做多。"""
    ma20 = df["close"].rolling(20).mean()
    signal = (df["close"] > ma20).astype(int)
    return signal.tolist()


def run():
    df = load_data("sh000300", recent_n=252)
    prices = df["close"].tolist()
    dates = df["date"].dt.strftime("%Y-%m-%d").tolist()

    signals = make_simple_signal(df)
    positions = signals_to_positions(signals)
    metrics = compute_metrics(prices, positions)

    out = {
        "symbol": "sh000300",
        "period": f"{dates[0]} ~ {dates[-1]}",
        "total_bars": len(prices),
        "signal_rule": "close > MA20 => long",
        "metrics": metrics,
        "recent_prices": list(zip(dates[-5:], prices[-5:])),
    }
    return out


def main():
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
