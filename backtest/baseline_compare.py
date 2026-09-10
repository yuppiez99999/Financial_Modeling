# backtest/baseline_compare.py
"""
预测 baseline 对比最小入口。

当前阶段目标:
- 先跑通一个最小对比流水线
- dumb 基准与简单模型基准共存
- 结果可留痕

后续可在此基础上接入 tsai 具体实现。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


def majority_baseline(y: Sequence[int]) -> List[int]:
    """
    构造 Majority class baseline 预测。
    """
    y = np.asarray(y, dtype=int)
    if len(y) == 0:
        return []
    counts = np.bincount(y)
    pred_class = int(np.argmax(counts))
    return [pred_class] * len(y)


def last_direction_baseline(
    prices: Sequence[float],
    horizon: int = 1,
) -> List[int]:
    """
    用最近方向作为预测基准。

    若最近价格上涨，则预测上涨；
    否则预测下跌或持平，视实现而定。
    """
    prices = np.asarray(prices, dtype=float)
    preds: List[int] = []
    for i in range(len(prices)):
        if i < horizon:
            preds.append(0)
            continue
        recent = prices[i] - prices[i - horizon]
        preds.append(1 if recent > 0 else 0)
    return preds


def constant_baseline(n: int, value: int = 1) -> List[int]:
    """
    构造恒定预测 baseline。
    """
    return [value] * n


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def run_comparison(
    y_true: Sequence[int],
    prices: Sequence[float],
) -> Dict[str, object]:
    """
    运行最小 baseline 对比。
    """
    baselines: Dict[str, List[int]] = {
        "majority": majority_baseline(list(y_true)),
        "last_direction": last_direction_baseline(list(prices)),
        "constant_up": constant_baseline(len(y_true), 1),
        "constant_down": constant_baseline(len(y_true), 0),
    }

    results: Dict[str, float] = {}
    for name, preds in baselines.items():
        results[name] = accuracy(y_true, preds)

    best_name = max(results, key=results.get)

    return {
        "baselines": baselines,
        "accuracy": results,
        "best_baseline": best_name,
        "best_accuracy": results[best_name],
    }


def make_toy_case() -> Dict[str, List[float]]:
    """
    构造一个用于演示的玩具案例。
    """
    return {
        "prices": [100.0, 101.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 107.0, 108.0],
        "y_true": [0, 1, 1, 0, 1, 1, 0, 1, 1, 1],
    }


def main() -> None:
    case = make_toy_case()
    result = run_comparison(case["y_true"], case["prices"])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
