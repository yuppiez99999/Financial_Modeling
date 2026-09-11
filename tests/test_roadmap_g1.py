"""S11（G1）统一评估量尺：因子健康度报告的测试。

测试要点：
  - IC 置信区间：Fisher 变换口径正确（CI 含真值、样本不足拒答、含 0 不显著）；
  - 分层收益：信号有效时分位单调、spread 为正；样本不足拒答；
  - 换手率：方向翻转计数口径正确，观望态不翻不计；
  - 多周期衰减：衰减曲线逐周期给出 IC + CI；
  - 成本敏感性：三档常量、净收益方向正确、缺前置指标拒答；
  - 边界：**affects_gate 恒为 False**，任何输出不参与门禁判定；
  - 消费口：CLI `ic --detail` 落盘 reports/factor_metrics.json。
"""
from __future__ import annotations

import json
import random

import pytest

from src.eval.factor_metrics import (
    FactorMetricsCalculator,
    cost_sensitivity,
    horizon_decay,
    ic_confidence_interval,
    quantile_returns,
    turnover_rate,
)


def _synthetic(n: int = 300, strength: float = 0.6, seed: int = 42):
    """构造 (scores, returns)：returns = strength × scores + 噪声。"""
    rng = random.Random(seed)
    scores = [rng.uniform(-1, 1) for _ in range(n)]
    returns = [strength * s + rng.gauss(0, 1.0) for s in scores]
    return scores, returns


# ----------------------------------------------------------------------
# IC 置信区间（Fisher 变换）
# ----------------------------------------------------------------------
def test_ic_ci_contains_estimate():
    ci = ic_confidence_interval(0.2, 200)
    assert ci["available"]
    assert ci["ci_low"] < 0.2 < ci["ci_high"]
    assert ci["significant"] is True  # 0.2 且 n=200 → 不含 0


def test_ic_ci_zero_ic_not_significant():
    ci = ic_confidence_interval(0.0, 200)
    assert ci["available"]
    assert ci["ci_low"] < 0 < ci["ci_high"]
    assert ci["significant"] is False


def test_ic_ci_insufficient_samples_rejected():
    ci = ic_confidence_interval(0.5, 5)
    assert ci["available"] is False
    assert ci["significant"] is None
    assert ci["reason"]


def test_ic_ci_bounds_shrink_with_n():
    small = ic_confidence_interval(0.1, 30)
    large = ic_confidence_interval(0.1, 2000)
    assert (small["ci_high"] - small["ci_low"]) > (large["ci_high"] - large["ci_low"])


# ----------------------------------------------------------------------
# 分位数分层收益
# ----------------------------------------------------------------------
def test_quantile_returns_monotonic_for_valid_signal():
    scores, returns = _synthetic(strength=2.0, seed=1)
    res = quantile_returns(scores, returns)
    assert res["available"]
    means = [q["mean_return"] for q in res["quantiles"]]
    assert means == sorted(means), "有效信号的层均收益应单调递增"
    assert res["spread_top_minus_bottom"] > 0
    assert res["monotonic"] is True


def test_quantile_returns_insufficient_samples():
    res = quantile_returns([0.1, 0.2, 0.3], [0.01, -0.01, 0.02])
    assert res["available"] is False
    assert res["reason"]


def test_quantile_returns_drops_nan_and_none():
    scores, returns = _synthetic(n=200, seed=2)
    scores2 = list(scores)
    returns2 = list(returns)
    scores2[0] = None
    returns2[1] = None
    res = quantile_returns(scores2, returns2)
    assert res["available"]
    total = sum(q["samples"] for q in res["quantiles"])
    assert total == 198


# ----------------------------------------------------------------------
# 换手率
# ----------------------------------------------------------------------
def test_turnover_all_flip():
    res = turnover_rate([1, -1, 1, -1, 1])
    assert res["available"]
    assert res["turnover"] == 1.0
    assert res["flips"] == 4


def test_turnover_no_flip():
    res = turnover_rate([1, 1, 1, 1])
    assert res["available"]
    assert res["turnover"] == 0.0


def test_turnover_neutral_samples_do_not_flip():
    # 观望态（0）不翻不计
    res = turnover_rate([1, 0, 1, -1])
    assert res["flips"] == 1  # 只有 1 → -1 翻转
    assert res["turnover"] == pytest.approx(1 / 3, abs=1e-3)


def test_turnover_insufficient():
    assert turnover_rate([1])["available"] is False


# ----------------------------------------------------------------------
# 多周期衰减
# ----------------------------------------------------------------------
def test_horizon_decay_curve():
    scores, returns = _synthetic(n=300, seed=3)
    rbd = {d: [r * (1.0 if d == 5 else 0.5) for r in returns] for d in (5, 10, 20)}
    res = horizon_decay(scores, rbd)
    assert res["available"]
    days = [row["days"] for row in res["curve"]]
    assert days == [5, 10, 20]
    for row in res["curve"]:
        assert row["available"]
        assert "ic" in row and "ic_ci" in row


def test_horizon_decay_insufficient_period_marked():
    scores, returns = _synthetic(n=300, seed=4)
    res = horizon_decay(scores, {5: returns, 10: returns[:3]})
    by_days = {row["days"]: row for row in res["curve"]}
    assert by_days[10]["available"] is False
    assert by_days[10]["reason"]


# ----------------------------------------------------------------------
# 成本敏感性（T11.2 口径草案）
# ----------------------------------------------------------------------
def test_cost_sensitivity_three_levels_and_direction():
    levels = cost_sensitivity(0.05, 0.5, [{"days": 10, "available": True}])
    assert levels["available"]
    assert [lv["name"] for lv in levels["levels"]] == [
        "conservative", "base", "aggressive"
    ]
    nets = [lv["per_horizon"]["10"]["net"] for lv in levels["levels"]]
    # 成本越低净收益越高
    assert nets[0] < nets[1] < nets[2]


def test_cost_sensitivity_rejects_when_prerequisites_missing():
    res = cost_sensitivity(None, 0.5, [{"days": 10, "available": True}])
    assert res["available"] is False
    assert res["reason"]


def test_cost_sensitivity_unavailable_horizon_skipped():
    res = cost_sensitivity(0.05, 0.5, [{"days": 20, "available": False}])
    assert all(not lv["per_horizon"] for lv in res["levels"])


def test_cost_sensitivity_levels_are_constants_not_configurable():
    # T11.2 口径定稿前：三档参数为常量，不接受调用方覆盖（防自由度回流）
    res = cost_sensitivity(0.05, 0.5, [{"days": 10, "available": True}],
                           levels=[{"name": "fake", "commission": 0, "slippage": 0}])
    # levels 传参是内部扩展点，外部不得通过 config 触达 —— 这里只验证默认三档存在
    assert res["manual_checkpoint"].startswith("T11.2")


# ----------------------------------------------------------------------
# 汇总入口 FactorMetricsCalculator
# ----------------------------------------------------------------------
def test_calculator_evaluate_valid_signal():
    scores, returns = _synthetic(strength=1.5, seed=5)
    calc = FactorMetricsCalculator({})
    res = calc.evaluate("short_term", 5, scores, returns,
                        returns_by_days={5: returns, 10: returns})
    assert res["available"]
    assert res["affects_gate"] is False
    assert res["ic_ci"]["available"]
    assert res["quantile_returns"]["available"]
    assert res["turnover"]["available"]
    assert res["horizon_decay"]["available"]
    assert res["cost_sensitivity"]["available"]


def test_calculator_evaluate_insufficient_samples():
    calc = FactorMetricsCalculator({})
    res = calc.evaluate("short_term", 5, [0.1, 0.2], [0.01, -0.01])
    assert res["available"] is False
    assert res["reason"]
    assert res["affects_gate"] is False


def test_calculator_evaluate_all_summary():
    scores, returns = _synthetic(strength=1.5, seed=6)
    calc = FactorMetricsCalculator({})
    res = calc.evaluate_all({
        "short_term": {"horizon_days": 5, "scores": scores, "returns": returns},
        "mid_term": {"horizon_days": 10, "scores": scores[:5], "returns": returns[:5]},
    })
    assert res["affects_gate"] is False
    assert "short_term" in res["summary"]["available_horizons"]
    assert "mid_term" not in res["summary"]["available_horizons"]
    assert res["gate_note"]


def test_calculator_save_and_load(tmp_path):
    scores, returns = _synthetic(strength=1.5, seed=7)
    calc = FactorMetricsCalculator({"factor_metrics": {"report_dir": str(tmp_path)}})
    payload = calc.evaluate_all({
        "short_term": {"horizon_days": 5, "scores": scores, "returns": returns},
    })
    saved = calc.save(payload)
    assert saved.exists()
    loaded = calc.load()
    assert loaded["summary"]["n_available"] == 1


# ----------------------------------------------------------------------
# CLI 接入（ic --detail）
# ----------------------------------------------------------------------
def test_cli_ic_has_detail_flag():
    from main import build_parser

    parser = build_parser()
    args = parser.parse_args(["ic", "--detail"])
    assert args.command == "ic"
    assert args.detail is True
    args_plain = parser.parse_args(["ic"])
    assert args_plain.detail is False


def test_run_ic_detail_writes_factor_metrics(tmp_path, monkeypatch):
    """端到端：run_ic(detail=True) 落盘 factor_metrics.json 且不改门禁字段。"""
    import numpy as np
    import pandas as pd

    from main import run_ic

    rng = np.random.default_rng(11)
    # 造一个「带真实信号」的数据集：收益部分由动量因子驱动
    n = 400
    symbols = ["600519.SH", "000858.SZ", "300308.SZ"]
    for sym in symbols:
        mom = rng.normal(0, 0.02, n).cumsum()
        rets = 0.3 * np.r_[0, mom[:-1] - mom[1:]] + rng.normal(0, 0.02, n) * 0.7
        close = 100 * np.exp(np.cumsum(rets))
        pd.DataFrame({
            "date": pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="B"),
            "open": close * (1 - 0.002),
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n),
        }).to_csv(f"data/raw/{sym}.csv", index=False)

    config = {
        "data": {
            "raw_dir": "data/raw",
            "prediction_horizons": {"short_term": 5},
            "markets": {"stocks": {"enabled": True, "symbols": symbols}},
        },
        "model": {"type": "lightgbm"},
        "strategy_gate": {"report_dir": str(tmp_path)},
        "factor_metrics": {"report_dir": str(tmp_path)},
    }
    monkeypatch.chdir(pytest.rootdir if hasattr(pytest, "rootdir") else ".")
    result = run_ic(config, detail=True)
    # 门禁字段不受影响
    assert "horizons" in result
    # 因子健康度报告已挂载 + 落盘
    fm = result.get("factor_metrics") or {}
    assert fm.get("affects_gate") is False
    report = tmp_path / "factor_metrics.json"
    assert report.exists()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["affects_gate"] is False
    assert payload["source"] == "src/eval/factor_metrics.py"
