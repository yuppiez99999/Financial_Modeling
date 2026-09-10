# backtest/baseline_real.py
"""基于真实数据的 baseline 对比。"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.data_provider import DataProvider
from backtest.run_minimal import compute_metrics, signals_to_positions


def load_prices(symbol: str = "sh000300", recent_n: int = 252):
    dp = DataProvider()
    df = dp.get_index_daily(symbol)
    if recent_n and recent_n < len(df):
        df = df.tail(recent_n).reset_index(drop=True)
    return df["close"].tolist()


def majority_baseline(n: int, value: int = 1):
    return [value] * n


def last_direction_baseline(prices, horizon: int = 1):
    preds = []
    for i in range(len(prices)):
        if i < horizon:
            preds.append(0)
        else:
            preds.append(1 if prices[i] > prices[i - horizon] else 0)
    return preds


def ma_cross_signal(prices, short: int = 5, long: int = 20):
    sma_short = pd.Series(prices).rolling(short).mean().bfill().values
    sma_long = pd.Series(prices).rolling(long).mean().bfill().values
    return (sma_short > sma_long).astype(int).tolist()


def run_comparison():
    prices = load_prices("sh000300", recent_n=252)
    n = len(prices)

    baselines = {
        "constant_long": majority_baseline(n, 1),
        "constant_short": majority_baseline(n, 0),
        "last_direction": last_direction_baseline(prices),
        "ma_cross_5_20": ma_cross_signal(prices, 5, 20),
    }

    results = {}
    for name, signals in baselines.items():
        positions = signals_to_positions(signals)
        metrics = compute_metrics(prices, positions)
        results[name] = metrics

    best = max(results, key=lambda k: results[k]["total_return"])

    return {
        "symbol": "sh000300",
        "total_bars": n,
        "baselines": results,
        "best_baseline": best,
        "best_total_return": results[best]["total_return"],
    }


def main():
    result = run_comparison()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
