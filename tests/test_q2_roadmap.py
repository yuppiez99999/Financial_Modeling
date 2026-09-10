"""Q2 排期：多因子集成 · IC/命中率策略门禁 · 前视 IC 计算 测试。

全部离线运行（不触网、不依赖已训练模型）；重点验证：
  - IC 计算的数学正确性与未到期样本处理；
  - 多因子加权的归一化/缺失因子 fail-soft；
  - 策略门禁的 fail-close 行为（缺数据不得默认放行）；
  - daily report 门禁章节的缺失/就绪两种分支。
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.factor_combiner import (
    DEFAULT_FACTOR_WEIGHTS,
    FactorCombiner,
    build_feature_factors,
    proba_to_score,
)
from src.inference.ic import (
    ICCalculator,
    forward_returns,
    hit_rate,
    spearman_ic,
)
from src.report.daily_report import DailyReportGenerator
from src.trading.gate import STATE_DISABLED, STATE_GATED, STATE_READONLY, StrategyGate


@pytest.fixture()
def cfg():
    with open(PROJECT_ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------- IC 计算 ----------------

def test_spearman_ic_perfect_and_inverse():
    assert spearman_ic([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert spearman_ic([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_spearman_ic_handles_ties_and_constant():
    # 并列值走平均秩，不因除零崩馈
    assert isinstance(spearman_ic([1, 1, 2, 2], [1, 2, 3, 4]), float)
    assert spearman_ic([1, 1, 1], [1, 2, 3]) == 0.0
    assert spearman_ic([1.0], [1.0]) == 0.0


def test_forward_returns_marks_undated_tail_as_none():
    out = forward_returns([100.0, 110.0, 121.0, 133.1], 2)
    assert out[0] == pytest.approx(0.21)
    assert out[1] == pytest.approx(0.21)
    # 末尾未到期样本必须是 None，绝不猜测
    assert out[2] is None and out[3] is None


def test_hit_rate_with_neutral_band():
    scores = [0.6, -0.5, 0.02, 0.4]
    returns = [0.01, -0.02, 0.03, -0.01]
    # 第 3 条 |score|<=0.05 视为观望，不计入分母 → 2/3
    assert hit_rate(scores, returns, neutral_band=0.05) == pytest.approx(2 / 3)
    assert hit_rate([], []) == 0.0


def test_ic_calculator_insufficient_samples_not_available():
    calc = ICCalculator(min_samples=30)
    res = calc.evaluate("short_term", 5, [0.1, 0.2], [0.01, 0.02])
    assert res.available is False
    assert res.passed is False
    assert "样本不足" in res.reason


def test_ic_calculator_positive_signal_passes():
    n = 200
    # 正相关且方向全对：高概率配正收益、低概率配负收益
    scores = [0.6 if i % 3 else 0.4 for i in range(n)]
    returns = [0.01 if i % 3 else -0.01 for i in range(n)]
    res = ICCalculator(min_samples=30, min_ic=0.03, min_hit_rate=0.5).evaluate(
        "long_term", 20, scores, returns, window_size=20
    )
    assert res.available is True
    assert res.passed is True
    assert res.ic > 0.9            # 强正相关
    assert res.hit_rate > 0.6      # 方向多数正确
    assert res.windows > 0


def test_ic_calculator_icir_zero_when_no_variance():
    scores = [0.6] * 100
    returns = [0.01] * 100
    res = ICCalculator(min_samples=30, min_ic=0.0, min_hit_rate=0.0).evaluate(
        "mid_term", 10, scores, returns, window_size=20
    )
    # 分数无方差 → IC 0，ICIR 不允许除零
    assert res.ic == 0.0
    assert res.icir == 0.0


def test_ic_evaluate_all_shape():
    calc = ICCalculator(min_samples=5, min_ic=0.0, min_hit_rate=0.0, min_windows=3)
    scores = [0.6, 0.4] * 5
    returns = [0.01, -0.01] * 5
    payload = {
        "short_term": {"horizon_days": 5, "scores": scores, "returns": returns},
        "mid_term": {"horizon_days": 10, "scores": scores, "returns": returns},
    }
    out = calc.evaluate_all(payload)
    assert set(out["horizons"].keys()) == {"short_term", "mid_term"}
    # 未指定 window_size 时不做 ICIR 窗口门槛（门禁只看 IC + 命中率）
    assert out["all_passed"] is True
    assert out["horizons"]["short_term"]["windows"] == 0


# ---------------- 多因子组合 ----------------

def test_proba_to_score_neutral_maps_to_zero():
    assert proba_to_score(0.5) == pytest.approx(0.0)
    assert proba_to_score(0.75) == pytest.approx(0.5)
    assert proba_to_score(0.25) == pytest.approx(-0.5)
    assert proba_to_score(None) == 0.0
    assert proba_to_score(float("nan")) == 0.0


def test_factor_combiner_weights_normalized():
    fc = FactorCombiner({})
    w = fc.resolve_weights(["lightgbm", "timesfm"])
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["lightgbm"] > w["timesfm"]


def test_factor_combiner_missing_factor_renormalizes():
    fc = FactorCombiner({})
    res = fc.combine("600519.SH", {"lightgbm": 0.5, "timesfm": None})
    assert res.missing == ["timesfm"]
    weights = {f.name: f.weight for f in res.factors}
    assert weights["timesfm"] == 0.0
    assert weights["lightgbm"] == pytest.approx(1.0)
    assert res.score == pytest.approx(0.5)
    assert res.direction == "看涨"


def test_factor_combiner_no_factor_missing_does_not_bias_null_as_zero():
    """缺失因子必须从权重中剔除（重归一化），而不是当作 0 分稀释得分。"""
    fc = FactorCombiner({})
    full = fc.combine("X", {"lightgbm": 1.0, "momentum": 1.0})
    partial = fc.combine("X", {"lightgbm": 1.0, "momentum": None})
    # 两者得分都应为满格 1.0；若把 None 当 0 分会得到 <1
    assert full.score == pytest.approx(1.0)
    assert partial.score == pytest.approx(1.0)


def test_factor_combiner_empty_weights_falls_back_to_equal():
    fc = FactorCombiner({"model_factors": {"weights": {"lightgbm": 0, "momentum": 0}}})
    res = fc.combine("X", {"lightgbm": 0.5, "momentum": 0.5})
    assert res.score == pytest.approx(0.5)


def test_factor_combiner_ic_weighter_scales_by_abs_ic():
    fc = FactorCombiner({"model_factors": {"weighter": "ic"}})
    w = fc.resolve_weights(["lightgbm", "momentum"], ic_by_factor={"lightgbm": 0.1, "momentum": 0.01})
    assert w["lightgbm"] > w["momentum"]
    assert sum(w.values()) == pytest.approx(1.0)


def test_factor_combiner_config_weights_override():
    fc = FactorCombiner({"model_factors": {"weights": {"lightgbm": 1.0, "momentum": 1.0}}})
    w = fc.resolve_weights(["lightgbm", "momentum"])
    assert w["lightgbm"] == pytest.approx(0.5)


def test_model_factor_scores_from_prediction():
    pred = {"probability": 0.7, "model": "lightgbm_short_term_5d"}
    out = FactorCombiner.model_factor_scores(pred)
    assert out["lightgbm"] == pytest.approx(0.4)
    assert FactorCombiner.model_factor_scores({"error": "x"}) == {}


def test_model_factor_scores_components():
    pred = {"components": {"lightgbm": 0.6, "timesfm": 0.4}}
    out = FactorCombiner.model_factor_scores(pred)
    assert out["lightgbm"] == pytest.approx(0.2)
    assert out["timesfm"] == pytest.approx(-0.2)


def test_build_feature_factors_on_synthetic_frame():
    n = 200
    # 前段横盘、后段加速上涨 → 近期动量处于滚动窗口高位，动量因子为正
    close = [100.0] * (n - 40) + [100.0 * (1.01 ** i) for i in range(1, 41)]
    df = pd.DataFrame({"close": close, "volume": [1000 + i for i in range(n)]})
    factors = build_feature_factors(df)
    assert set(factors.keys()) == {"momentum", "trend", "volume", "volatility"}
    assert all(-1.0 <= v <= 1.0 for v in factors.values())
    assert factors["momentum"] > 0   # 近期动量为窗口高位
    assert factors["trend"] > 0      # 收盘价高于 MA20
    assert factors["volume"] > 0     # 近期放量


def test_build_feature_factors_fail_soft_on_empty():
    assert all(v == 0.0 for v in build_feature_factors(None).values())
    assert all(v == 0.0 for v in build_feature_factors(pd.DataFrame()).values())


def test_default_weights_reference_config_keys(cfg):
    """配置文件中的因子名必须都有默认权重兜底。"""
    configured = set((cfg.get("model_factors", {}) or {}).get("weights", {}))
    assert configured, "config 必须定义 model_factors.weights"
    assert configured.issubset(set(DEFAULT_FACTOR_WEIGHTS))


# ---------------- 策略门禁 ----------------

def _ic_payload(passed_all=True):
    return {
        "horizons": {
            "short_term": {"passed": passed_all, "ic": 0.05, "icir": 0.2,
                           "hit_rate": 0.55, "samples": 100, "windows": 5,
                           "reason": "" if passed_all else "命中率 51% < 52%"},
            "mid_term": {"passed": passed_all, "ic": 0.04, "icir": 0.2,
                         "hit_rate": 0.54, "samples": 100, "windows": 5,
                         "reason": "" if passed_all else "命中率 51% < 52%"},
        }
    }


def test_gate_fail_close_on_missing_data():
    d = StrategyGate({}).decide({})
    assert d.state == STATE_READONLY
    assert d.passed is False
    # fail-close：不得因缺数据而放行，且必须给出原因
    assert d.blocked_by
    assert "IC" in d.blocked_by[0]


def test_gate_readonly_when_partial_pass_with_scope_all():
    d = StrategyGate({}).decide(_ic_payload(passed_all=False))
    assert d.state == STATE_READONLY
    assert len(d.blocked_by) == 2


def test_gate_gated_when_all_pass():
    d = StrategyGate({}).decide(_ic_payload(passed_all=True))
    assert d.state == STATE_GATED
    assert d.passed is True


def test_gate_scope_any():
    payload = {
        "horizons": {
            "short_term": {"passed": False, "ic": 0.0, "hit_rate": 0.5, "reason": "未达标"},
            "long_term": {"passed": True, "ic": 0.06, "hit_rate": 0.55, "reason": ""},
        }
    }
    assert StrategyGate({"strategy_gate": {"scope": "any"}}).decide(payload).state == STATE_GATED
    assert StrategyGate({"strategy_gate": {"scope": "all"}}).decide(payload).state == STATE_READONLY


def test_gate_force_readonly_overrides_pass():
    d = StrategyGate({"strategy_gate": {"force_readonly": True}}).decide(_ic_payload(True))
    assert d.state == STATE_READONLY
    assert d.blocked_by == ["force_readonly"]


def test_gate_disabled_state():
    d = StrategyGate({"strategy_gate": {"enabled": False}}).decide(_ic_payload(True))
    assert d.state == STATE_DISABLED
    assert d.passed is False


def test_gate_audit_cross_check_blocks():
    cfg = {"strategy_gate": {"use_audit": True}}
    d = StrategyGate(cfg).decide(_ic_payload(True), {"available": True, "verified": 5, "hit_rate": 0.3})
    assert d.passed is False
    assert any("审计" in x for x in d.blocked_by)


def test_gate_audit_cross_check_passes():
    cfg = {"strategy_gate": {"use_audit": True}}
    d = StrategyGate(cfg).decide(_ic_payload(True), {"available": True, "verified": 50, "hit_rate": 0.6})
    assert d.passed is True
    assert d.horizons["_audit"]["verified"] == 50


def test_gate_evaluate_from_audit_records():
    records = [
        {"horizon": "short_term", "verified": True, "hit": True},
        {"horizon": "short_term", "verified": True, "hit": False},
        {"horizon": "mid_term", "verified": False, "hit": None},
    ]
    stats = StrategyGate().evaluate_from_audit_records(records)
    assert stats["verified"] == 2
    assert stats["hit_rate"] == pytest.approx(0.5)
    assert stats["by_horizon"]["short_term"]["verified"] == 2


def test_gate_decision_serializable():
    payload = StrategyGate({}).decide(_ic_payload(True)).to_dict()
    json.dumps(payload, ensure_ascii=False)  # 必须可 JSON 序列化
    assert payload["generated_at"]


def test_config_has_q2_sections(cfg):
    assert "strategy_gate" in cfg
    assert "model_factors" in cfg
    assert cfg["strategy_gate"]["scope"] in {"all", "any"}
    assert 0 < cfg["strategy_gate"]["min_hit_rate"] < 1


# ---------------- 日报门禁章节 ----------------

def test_daily_report_gate_section_missing_file(tmp_path, cfg):
    cfg["report"]["output_dir"] = str(tmp_path / "reports")
    cfg["strategy_gate"] = {"report_dir": str(tmp_path / "reports")}
    lines = DailyReportGenerator(cfg)._generate_gate_section()
    text = "\n".join(lines)
    assert "策略门禁" in text
    assert "未评估" in text
    assert "只读" in text


def test_daily_report_gate_section_renders(tmp_path, cfg):
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "strategy_gate.json").write_text(
        json.dumps(StrategyGate({"strategy_gate": {"scope": "any"}}).decide(_ic_payload(True)).to_dict(),
                   ensure_ascii=False),
        encoding="utf-8",
    )
    cfg["report"]["output_dir"] = str(reports)
    cfg["strategy_gate"] = {"report_dir": str(reports)}
    text = "\n".join(DailyReportGenerator(cfg)._generate_gate_section())
    assert "gated" in text
    assert "short_term" in text


# ---------------- CLI 接线 ----------------

def test_cli_has_q2_commands():
    import main as main_cli

    parser = main_cli.build_parser()
    for cmd in ("ic", "gate", "factors"):
        assert parser.parse_args([cmd, "600519.SH"]).command == cmd


def test_cli_config_symbols_dedup():
    import main as main_cli

    cfg = {"data": {"markets": {"stock": {"enabled": True, "symbols": ["A", "B"]},
                                "etf": {"enabled": True, "symbols": ["B", "C"]},
                                "futures": {"enabled": False, "symbols": ["X"]}}}}
    assert main_cli._config_symbols(cfg) == ["A", "B", "C"]
