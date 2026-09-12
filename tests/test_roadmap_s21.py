"""S21 组合回测闭环守卫：引擎会计正确 + 无未来函数 + 成本口径 + 一致性对照。

全部离线合成数据（不触网、不依赖模型）。聚焦四件事：
  1. **无未来函数**（T21.1 的生死线）：T 日收盘决定的权重绝不能影响 T 日
     收益 —— 引擎必须等价于 shift(1) 持仓；
  2. 成本扣减与 T11.2 三档口径一致（单一事实源 factor_metrics.T112_COST_TIERS）；
  3. 等权归一与杠杆禁令；
  4. T21.2 一致性对照：IC/命中率 vs 组合 PnL 的背离能被如实识别。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.factor_metrics import T112_COST_TIERS, cost_sensitivity  # noqa: E402
from src.eval.portfolio_backtest import (  # noqa: E402
    build_equal_weight_plan,
    consistency_contrast,
    cost_one_side,
    run_portfolio_backtest,
)


# ---------------------------------------------------------------- helpers
def _price_frame(values: list[float], start: str = "2026-01-01") -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.date_range(start, periods=len(values), freq="D"),
        "close": values,
    })


def _plan(values: list[float], start: str = "2026-01-01") -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.date_range(start, periods=len(values), freq="D"),
        "target_weight": values,
    })


def _signals(actions: list[str], confs: list[float], start: str = "2026-01-01") -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.date_range(start, periods=len(actions), freq="D"),
        "action": actions,
        "confidence": confs,
    })


# ---------------------------------------------------------------- T112 口径
def test_t112_cost_tiers_are_the_single_source():
    """三档单边成本必须与 T11.2 定稿值一致：0.125% / 0.075% / 0.030%。"""
    assert cost_one_side("conservative") == pytest.approx(0.00125)
    assert cost_one_side("base") == pytest.approx(0.00075)
    assert cost_one_side("aggressive") == pytest.approx(0.00030)
    with pytest.raises(ValueError):
        cost_one_side("custom-cheap")  # 定稿档之外不存在「更便宜的一档」


def test_cost_sensitivity_uses_shared_tiers():
    """cost_sensitivity 缺省档与共享常量同源（防两处口径漂移）。"""
    out = cost_sensitivity(quantile_spread=0.01, turnover=0.2,
                           horizons=[{"days": 5, "available": True}])
    assert out["available"] is True
    assert [lv["name"] for lv in out.get("levels", [])] == \
        ["conservative", "base", "aggressive"]


# ---------------------------------------------------------------- 无未来函数
def test_no_lookahead_weight_decided_today_cannot_earn_today():
    """T 日收盘才决定建仓，T 日的大涨绝不能进收益（shift(1) 生死线）。"""
    # 价格：D6 单日 +10%；权重 D6 收盘才变 1 —— 若引擎吃当日权重即作弊
    prices = _price_frame([100, 100, 100, 100, 100, 110])
    plan = _plan([0, 0, 0, 0, 0, 1])
    bt = run_portfolio_backtest({"S": prices}, {"S": plan},
                                cost_one_side_value=0.0, normalize=None)
    daily = bt["daily"]
    assert float(daily.loc[5, "net"]) == 0.0  # 跳涨日建仓：当日不得吃到 +10%


def test_no_lookahead_exit_day_still_earns_move():
    """权重在 D6 清零：D6 的涨跌仍属于 D5 建的仓（持仓按 T-1 权重计）。"""
    prices = _price_frame([100, 100, 100, 100, 100, 110, 110])
    plan = _plan([0, 0, 0, 0, 1, 0, 0])
    bt = run_portfolio_backtest({"S": prices}, {"S": plan},
                                cost_one_side_value=0.0, normalize=None)
    daily = bt["daily"]
    assert float(daily.loc[5, "net"]) == pytest.approx(0.10, abs=1e-9)
    assert float(daily.loc[6, "net"]) == 0.0


# ---------------------------------------------------------------- 成本会计
def test_cost_is_turnover_times_one_side():
    """成本 = 换手 × 单边成本：首日建仓计一次，之后空转零成本。"""
    prices = _price_frame([100, 100, 100])
    plan = _plan([0, 1, 1])
    bt = run_portfolio_backtest({"S": prices}, {"S": plan},
                                cost_one_side_value=0.001, normalize=None)
    cost = bt["daily"]["cost"]
    # idx0：目标权重 0（无建仓）→ 零成本；idx1：建仓 1 → 换手 1 → 成本 0.001
    assert float(cost.iloc[0]) == 0.0
    assert float(cost.iloc[1]) == pytest.approx(0.001, abs=1e-12)
    assert float(cost.iloc[2]) == 0.0
    assert bt["metrics"]["total_cost_drag"] == pytest.approx(0.001)


def test_buy_and_hold_zero_cost_when_free():
    prices = _price_frame([100.0 * (1 + 0.001 * i) for i in range(30)])
    plan = _plan([1] * 30)
    bt = run_portfolio_backtest({"S": prices}, {"S": plan},
                                cost_one_side_value=0.0, normalize=None)
    assert bt["metrics"]["total_return"] == pytest.approx(
        prices["close"].iloc[-1] / prices["close"].iloc[0] - 1.0, rel=1e-6)
    assert bt["metrics"]["sharpe"] is not None


# ---------------------------------------------------------------- 归一与杠杆
def test_equal_weight_normalization_splits_full_capital():
    """两只同时激活 → 各 0.5；一只激活 → 独占满仓（等权归一逐日重算）。"""
    prices = {s: _price_frame([100, 101, 102, 103]) for s in ("A", "B")}
    plans = {"A": _plan([1, 1, 1, 1]), "B": _plan([1, 1, 0, 0])}
    bt = run_portfolio_backtest(prices, plans, cost_one_side_value=0.0,
                                normalize="equal_active")
    daily = bt["daily"]
    r1 = 101 / 100 - 1   # idx1 的收益（持仓 = idx0 权重）
    r3 = 103 / 102 - 1   # idx3 的收益（持仓 = idx2 权重）
    # idx1：两只在 idx0 都激活 → 各半（daily 输出按 1e-8 舍入）
    assert float(daily.loc[1, "net"]) == pytest.approx(0.5 * r1 + 0.5 * r1, abs=1e-8)
    # idx3：idx2 只有 A 激活 → A 独占满仓
    assert float(daily.loc[3, "net"]) == pytest.approx(r3, abs=1e-8)
    assert bt["metrics"]["n_active_days"] >= 1


def test_leverage_is_rejected():
    """总权重 > 1（杠杆）必须 fail-close 拒绝，而不是静默缩放。"""
    prices = {"A": _price_frame([100, 101, 102])}
    plans = {"A": _plan([1.5, 1.0, 1.0])}
    bt = run_portfolio_backtest(prices, plans, normalize=None)
    assert bt["available"] is False
    assert "杠杆" in bt["reason"] or "> 1" in bt["reason"]


def test_common_dates_with_missing_prices_is_unavailable():
    """共同交易日内价格缺失（如停牌）必须 fail-close，不得静默按 0 收益。"""
    prices = {"A": _price_frame([100, 101, 102]),
              "B": _price_frame([100, 101, 102, 103], start="2026-02-01")}
    plans = {s: _plan([1] * len(prices[s])) for s in prices}
    bt = run_portfolio_backtest(prices, plans)
    assert bt["available"] is False
    assert "价格缺失" in bt["reason"]


def test_plan_symbol_without_price_is_unavailable():
    bt = run_portfolio_backtest(
        {"A": _price_frame([100, 101])}, {"GHOST": _plan([1, 1])})
    assert bt["available"] is False
    assert "GHOST" in bt["reason"]


# ---------------------------------------------------------------- 信号→计划
def test_build_equal_weight_plan_filters_by_confidence():
    sig = _signals(["BUY", "BUY", "HOLD", "SELL"], [0.9, 0.1, 0.9, 0.9])
    plans = build_equal_weight_plan({"S": sig}, min_confidence=0.5)
    w = plans["S"]["target_weight"].tolist()
    assert w == [1.0, 0.0, 0.0, 0.0]


def test_build_equal_weight_plan_accepts_numeric_action():
    sig = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=3),
        "action": [1, 0, -1],
        "confidence": [1.0, 1.0, 1.0],
    })
    plans = build_equal_weight_plan({"S": sig})
    assert plans["S"]["target_weight"].tolist() == [1.0, 0.0, 0.0]


# ---------------------------------------------------------------- 一致性对照
def test_consistency_contrast_flags_divergence():
    """构造「命中率 > 50% 但扣费后亏损」的标的 → 背离清单必须点名。"""
    # 高胜率小赚 + 一次大亏：命中率 2/3 但净收益为负（成本前就亏）
    prices_a = _price_frame([100, 101, 102, 90, 91])
    plan_a = _plan([0, 1, 1, 1, 1])
    # 稳定小赚：命中率 100%，净收益为正
    prices_b = _price_frame([100, 101, 102, 103, 104])
    plan_b = _plan([0, 1, 1, 1, 1])
    prices_c = _price_frame([50, 50.5, 51, 51.5, 52])
    plan_c = _plan([0, 1, 1, 1, 1])
    out = consistency_contrast(
        {"A": prices_a, "B": prices_b, "C": prices_c},
        {"A": plan_a, "B": plan_b, "C": plan_c},
        cost_one_side_value=0.0001)
    assert out["available"] is True
    rows = {r["symbol"]: r for r in out["rows"]}
    assert rows["A"]["hit_rate"] == pytest.approx(2 / 3, abs=1e-6)
    assert rows["A"]["net_return"] < 0
    divergent_symbols = {d["symbol"] for d in out["divergent_symbols"]}
    assert "A" in divergent_symbols
    assert "B" not in divergent_symbols


def test_consistency_contrast_needs_three_symbols():
    out = consistency_contrast(
        {"A": _price_frame([100, 101, 102]), "B": _price_frame([100, 101, 102])},
        {"A": _plan([1, 1, 1]), "B": _plan([1, 1, 1])},
        cost_one_side_value=0.0)
    assert out["available"] is False
    assert "可对照标的不足" in out["reason"]


# ---------------------------------------------------------------- 确定性
def test_engine_is_deterministic():
    prices = {s: _price_frame([100, 101, 100.5, 102, 103, 101, 104]) for s in ("A", "B")}
    plans = {"A": _plan([1, 1, 0, 1, 1, 0, 1]), "B": _plan([0, 1, 1, 1, 0, 0, 1])}
    a = run_portfolio_backtest(prices, plans, cost_one_side_value=0.00075)
    b = run_portfolio_backtest(prices, plans, cost_one_side_value=0.00075)
    assert a["daily"].equals(b["daily"])
    assert a["metrics"] == b["metrics"]
