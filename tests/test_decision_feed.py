"""决策源契约适配层守卫（src/export/decision_feed.py）。

全部离线纯函数，不触网、不训练。聚焦三件事：
  1. **入口口径唯一化**：net_up_probability 必须消除「方向 ↔ 概率」口径分裂，
     非有限值 / 缺失一律 None（**绝不返回 0.5** 冒充中性读数）；
  2. **聚合如实**：缺失周期不补 0.5；权重在可用周期内重新归一；
     三周期全缺 → available=false 而不是吐一个 0.5；
  3. **纪律字段**：advisory_only / affects_gate=false / position_role=observer
     必须结构性存在（下游据此保证「只读」不被悄悄破坏）。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.export import decision_feed as df  # noqa: E402


# ---------------------------------------------------------------- 净看涨概率
def test_net_up_probability_removes_direction_ambiguity():
    """看跌 + 概率 0.7 必须读成 净看涨 0.3 —— 下游不再需要判断 direction。"""
    assert df.net_up_probability("看涨", 0.7) == 0.7
    assert df.net_up_probability("看跌", 0.7) == 0.3
    # 两个方向各自 0.7 的概率必须给出相反结论（这是口径分裂的根治点）
    assert df.net_up_probability("看涨", 0.7) + df.net_up_probability("看跌", 0.7) == 1.0


def test_net_up_probability_refuses_to_fake_neutral():
    """非有限 / 缺失一律 None：0.5 是有效中性读数，不能与"数据坏了"混同。"""
    for bad in (None, float("nan"), float("inf"), "-inf", "abc", object()):
        assert df.net_up_probability("看涨", bad) is None, bad
    # 真 0.5 仍然是合法读数
    assert df.net_up_probability("看涨", 0.5) == 0.5


def test_net_up_probability_clamps_out_of_range():
    assert df.net_up_probability("看涨", 1.7) == 1.0
    assert df.net_up_probability("看涨", -0.3) == 0.0


# ---------------------------------------------------------------- 聚合
def _hz(direction="看涨", p=0.7, model="m"):
    return {"direction": direction, "probability": p, "model": model}


def test_aggregate_missing_horizon_is_not_backfilled_with_half():
    """缺失周期**不补 0.5**：两个周期全看多时综合分必须明显 > 0.5。"""
    agg = df.aggregate_horizons({
        "short_term": _hz("看涨", 0.9),
        "mid_term": _hz("看涨", 0.9),
        # long_term 缺失
    })
    assert agg["available"] is True
    assert agg["composite_score"] == 0.9
    assert agg["missing_horizons"] == ["long_term"]
    assert math.isclose(agg["coverage"], (0.30 + 0.35) / 1.0)


def test_aggregate_equal_weight_opposite_directions_net_out():
    """同权重同幅度的相反方向必须净额中性（不被方向字符串主导）。"""
    agg = df.aggregate_horizons(
        {"short_term": _hz("看涨", 0.8), "mid_term": _hz("看跌", 0.8)},
        weights={"short_term": 0.5, "mid_term": 0.5})
    assert agg["available"] is True
    assert math.isclose(agg["composite_score"], 0.5, abs_tol=1e-9)
    assert math.isclose(agg["composite_signed"], 0.0, abs_tol=1e-9)


def test_aggregate_weights_are_explicit_not_implicit():
    """综合分必须**按声明权重**加权：0.3×0.8 + 0.35×0.2 → 0.476923。"""
    agg = df.aggregate_horizons({
        "short_term": _hz("看涨", 0.8),
        "mid_term": _hz("看跌", 0.8),
    })
    assert math.isclose(agg["composite_score"], 0.476923, abs_tol=1e-6)
    assert agg["weights_used"] == {"short_term": 0.30, "mid_term": 0.35}


def test_aggregate_all_missing_is_unavailable_not_neutral():
    agg = df.aggregate_horizons({})
    assert agg["available"] is False
    assert "composite_score" not in agg
    assert agg["missing_horizons"] == ["short_term", "mid_term", "long_term"]


def test_aggregate_error_entry_treated_as_missing():
    agg = df.aggregate_horizons({
        "short_term": {"error": "模型未加载"},
        "mid_term": _hz("看涨", 0.6),
    })
    assert agg["missing_horizons"] == ["short_term", "long_term"]
    assert agg["composite_score"] == 0.6


# ---------------------------------------------------------------- 置信度 / 门槛
def test_confidence_uses_the_single_repo_wide_convention():
    assert df.confidence_from_score(0.5) == 0.0
    assert df.confidence_from_score(1.0) == 1.0
    assert df.confidence_from_score(0.25) == 0.5


def test_advisory_verdict_is_structurally_readonly():
    v_ok = df.advisory_verdict(0.4, 0.2)
    v_no = df.advisory_verdict(0.05, 0.2)
    assert v_ok["advisory_consumable"] is True
    assert v_no["advisory_consumable"] is False
    for v in (v_ok, v_no):
        assert v["advisory_only"] is True
        assert v["affects_gate"] is False


# ---------------------------------------------------------------- 完整契约
def _raw_payload():
    return {
        "generated_at": "2026-09-16T08:00:00",
        "model_type": "lightgbm",
        "predictions": [
            {"symbol": "600519.SH", "sector": "食品饮料", "horizons": {
                "short_term": _hz("看涨", 0.72, "lightgbm_short_term_5d"),
                "mid_term": _hz("看跌", 0.61, "lightgbm_mid_term_10d"),
                "long_term": _hz("看涨", 0.58, "lightgbm_long_term_20d"),
            }},
            {"symbol": "BAD", "sector": "", "horizons": {
                "short_term": {"error": "无数据"}}},
        ],
        "meta": {"models_loaded": 3, "symbol_count": 2, "error_symbols": []},
    }


def test_build_decision_feed_keeps_original_fields():
    """向后兼容：原始 direction/probability/model 逐字段保留，只新增。"""
    feed = df.build_decision_feed(_raw_payload(), {})
    p = feed["predictions"][0]
    assert p["horizons"]["short_term"]["direction"] == "看涨"
    assert p["horizons"]["short_term"]["probability"] == 0.72
    assert p["horizons"]["short_term"]["model"] == "lightgbm_short_term_5d"
    assert p["horizons"]["short_term"]["net_up_probability"] == 0.72
    assert p["horizons"]["mid_term"]["net_up_probability"] == 0.39


def test_build_decision_feed_orders_no_position_and_no_gate_change():
    feed = df.build_decision_feed(_raw_payload(), {})
    assert feed["position_role"] == "observer"
    assert feed["role"] == "decision_source_readonly"
    assert feed["advisory_config"]["affects_gate"] is False
    assert feed["advisory_config"]["advisory_only"] is True
    assert feed["meta"]["affects_gate"] is False
    assert feed["meta"]["readonly"] is True


def test_build_decision_feed_counts_and_error_symbols():
    feed = df.build_decision_feed(_raw_payload(), {})
    assert feed["meta"]["symbol_count"] == 2
    assert feed["meta"]["scored_count"] == 1          # BAD 无可用周期，不计分
    assert feed["predictions"][1]["aggregate"]["available"] is False
    assert feed["predictions"][1]["advisory"]["advisory_consumable"] is False


def test_build_decision_feed_never_raises_on_garbage():
    """垃圾输入必须产出可用结构（fail-soft），不得抛异常拖垮端点。"""
    for bad in ({}, {"predictions": None}, {"predictions": [None]}):
        feed = df.build_decision_feed(bad, {})
        assert feed["contract_version"] == df.FEED_VERSION
        assert isinstance(feed["predictions"], list)


def test_threshold_is_configurable_and_reported():
    feed = df.build_decision_feed(_raw_payload(),
                                  {"decision_feed": {"advisory_threshold": 0.35}})
    assert feed["advisory_config"]["recommended_threshold"] == 0.35
    assert feed["predictions"][0]["advisory"]["recommended_threshold"] == 0.35
