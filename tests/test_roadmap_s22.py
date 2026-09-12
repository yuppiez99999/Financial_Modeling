"""S22 三臂对照守卫：等权 / 1-ATR / 置信度加权（T22.1 + T22.2）。

全部离线合成数据。聚焦：
  1. `compute_atr_series` 与 `compute_atr` 同口径（末值相等）——单一 ATR 事实源；
  2. 1/ATR 臂：低波动标的权重更大、激活内 Σ=1、warmup 期零权重、无杠杆；
  3. 置信度臂：恒定置信度**结构性退化为等权**（与等权臂逐日净值一致）；
  4. 三臂计划都能被引擎执行且确定性可复现。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.indicators import compute_atr, compute_atr_series  # noqa: E402
from src.eval.portfolio_backtest import (  # noqa: E402
    build_atr_inverse_plan,
    build_confidence_plan,
    build_equal_weight_plan,
    run_portfolio_backtest,
)


# ---------------------------------------------------------------- helpers
def _ohlcv(closes, start="2026-01-01", amp=0.5):
    closes = list(closes)
    n = len(closes)
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "date": pd.date_range(start, periods=n),
        "open": closes * 0.999,
        "high": closes + amp,
        "low": closes - amp,
        "close": closes,
    })


def _plan01(values, start="2026-01-01"):
    return pd.DataFrame({
        "date": pd.date_range(start, periods=len(values)),
        "target_weight": values,
    })


# ---------------------------------------------------------------- ATR 口径
def test_atr_series_matches_compute_atr():
    """逐日序列的末值必须等于 compute_atr（同一 Wilder 公式，两处出口）。"""
    df = _ohlcv([100 + 0.5 * np.sin(i / 3.0) for i in range(60)])
    series = compute_atr_series(df, window=14)
    assert series is not None and len(series) == len(df)
    assert float(series.iloc[-1]) == pytest.approx(compute_atr(df, window=14))


def test_atr_series_refuses_insufficient_sample():
    assert compute_atr_series(_ohlcv([100, 101]), window=14) is None
    assert compute_atr_series(_ohlcv([100, 101, 102]).drop(columns=["low"])) is None


# ---------------------------------------------------------------- 1/ATR 臂
def test_atr_inverse_prefers_lower_volatility_and_sums_to_one():
    """低波动标的应拿到更大权重；激活日 Σ=1；warmup（前 13 天）零权重。"""
    calm = _ohlcv([100 + 0.1 * i for i in range(60)], amp=0.05)   # 低波动
    wild = _ohlcv([100 + 2.0 * np.sin(i / 3.0) for i in range(60)], amp=2.0)  # 高波动
    price_frames = {"CALM": calm, "WILD": wild}
    active = {"CALM": _plan01([1] * 60), "WILD": _plan01([1] * 60)}
    plans = build_atr_inverse_plan(price_frames, active, atr_window=14)

    w_calm = plans["CALM"]["target_weight"]
    w_wild = plans["WILD"]["target_weight"]
    # warmup：前 atr_window-1 天无 ATR 权重（与 compute_atr 拒绝口径一致）
    assert (w_calm[:13] == 0).all() and (w_wild[:13] == 0).all()
    # 激活日两只都为 1 → 归一后 Σ=1
    total = w_calm[13:].to_numpy() + w_wild[13:].to_numpy()
    assert np.allclose(total, 1.0, atol=1e-12)
    # 低波动者权重更大（两只都是等差/正弦路径，CALM 波幅显著更小）
    assert (w_calm[13:].to_numpy() > w_wild[13:].to_numpy()).all()
    assert (plans["CALM"]["target_weight"].between(0, 1)).all()


def test_atr_inverse_plan_is_executable_by_engine():
    calm = _ohlcv([100 + 0.1 * i for i in range(60)], amp=0.05)
    wild = _ohlcv([100 + 2.0 * np.sin(i / 3.0) for i in range(60)], amp=2.0)
    price_frames = {"CALM": calm, "WILD": wild}
    active = {"CALM": _plan01([1] * 60), "WILD": _plan01([1] * 60)}
    plans = build_atr_inverse_plan(price_frames, active)
    bt = run_portfolio_backtest(price_frames, plans,
                                cost_one_side_value=0.00075, normalize=None)
    assert bt["available"] is True
    assert bt["metrics"]["total_cost_drag"] >= 0.0
    # 只在 warmup 结束后建仓 → 建仓换手发生在第 14 天（idx 13）
    daily = bt["daily"]
    assert float(daily.loc[0, "cost"]) == 0.0
    assert float(daily.loc[13, "cost"]) > 0.0


def test_atr_inverse_requires_high_low_columns():
    calm = _ohlcv([100, 101, 102] * 10).drop(columns=["high", "low"])
    with pytest.raises(ValueError, match="high/low"):
        build_atr_inverse_plan(
            {"CALM": calm}, {"CALM": _plan01([1] * 30)})


# ---------------------------------------------------------------- 置信度臂
def test_confidence_arm_degenerates_to_equal_weight_for_constant_confidence():
    """恒定置信度（机械基线）→ 置信度臂与等权臂逐日净值完全一致。"""
    prices = {s: _ohlcv([100, 101, 100.5, 102, 103, 101, 104, 105]) for s in ("A", "B")}
    signals = {s: pd.DataFrame({
        "date": prices[s]["date"],
        "action": ["BUY"] * 8,
        "confidence": [1.0] * 8,
    }) for s in prices}
    active = build_equal_weight_plan(signals)
    conf_plans = build_confidence_plan(signals, active)

    bt_equal = run_portfolio_backtest(prices, active,
                                      cost_one_side_value=0.00075,
                                      normalize="equal_active")
    bt_conf = run_portfolio_backtest(prices, conf_plans,
                                     cost_one_side_value=0.00075,
                                     normalize=None)
    assert bt_equal["available"] and bt_conf["available"]
    assert bt_equal["metrics"] == bt_conf["metrics"]


def test_confidence_arm_is_proportional_to_confidence():
    """conf A=0.9 / B=0.1 且都激活 → A 拿 90% 仓位。"""
    dates = pd.date_range("2026-01-01", periods=3)
    signals = {
        "A": pd.DataFrame({"date": dates, "action": ["BUY"] * 3, "confidence": [0.9] * 3}),
        "B": pd.DataFrame({"date": dates, "action": ["BUY"] * 3, "confidence": [0.1] * 3}),
    }
    active = build_equal_weight_plan(signals)
    plans = build_confidence_plan(signals, active)
    assert plans["A"]["target_weight"].iloc[0] == pytest.approx(0.9, abs=1e-12)
    assert plans["B"]["target_weight"].iloc[0] == pytest.approx(0.1, abs=1e-12)


# ---------------------------------------------------------------- 三臂可执行
def test_all_arms_deterministic_and_executable():
    closes = [100 + 0.8 * np.sin(i / 4.0) + 0.05 * i for i in range(80)]
    prices = {"A": _ohlcv(closes, amp=0.4), "B": _ohlcv(closes, amp=1.5)}
    signals = {
        s: pd.DataFrame({
            "date": prices[s]["date"],
            "action": ["BUY" if i % 3 else "HOLD" for i in range(80)],
            "confidence": [0.6 + 0.1 * (i % 4) for i in range(80)],
        }) for s in ("A", "B")}
    active = build_equal_weight_plan(signals, min_confidence=0.0)
    arms = {
        "equal": (active, "equal_active"),
        "atr_inverse": (build_atr_inverse_plan(prices, active), None),
        "confidence": (build_confidence_plan(signals, active), None),
    }
    snapshots = {}
    for name, (plans, norm) in arms.items():
        bt = run_portfolio_backtest(prices, plans,
                                    cost_one_side_value=0.00075, normalize=norm)
        assert bt["available"] is True, f"{name} 臂不可执行"
        assert bt["metrics"]["max_drawdown"] <= 0.0
        snapshots[name] = bt["daily"]
    # 复跑一致
    bt2 = run_portfolio_backtest(prices, arms["equal"][0],
                                 cost_one_side_value=0.00075,
                                 normalize=arms["equal"][1])
    assert snapshots["equal"].equals(bt2["daily"])
