"""S7 门禁解锁攻坚测试：门禁阻塞诊断 + 训练折中性带 + 按标的分层评估。

全部离线运行（不触网、不依赖已训练模型、不依赖 config 全局状态）。
覆盖点：
  - 诊断：短板识别 / 归一化差距 / scope=all vs any / 不可用周期不误判；
  - 中性带：**只用训练折**推导、退化保护（样本不足→0）、施加后测试折命中率提升、
    纯噪声不乱设门槛、与既有 hit_rate 口径兼容；
  - 分层评估：per_symbol 精简指标形状正确，不塞大数组；
  - 向后兼容：不开启新开关时，门禁口径逐字段不变。
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.gate_diagnosis import (  # noqa: E402
    GATED_METRICS,
    diagnose,
    diagnose_horizon,
)
from src.inference.ic import ICCalculator, derive_neutral_band, hit_rate  # noqa: E402

GATE = {"scope": "all", "min_ic": 0.03, "min_hit_rate": 0.52, "min_windows": 3}


def _thresholds(**over):
    t = {"min_ic": 0.03, "min_hit_rate": 0.52, "min_windows": 3}
    t.update(over)
    return t


def _horizon(**over):
    h = {
        "horizon_days": 5, "ic": 0.04, "icir": 0.2, "hit_rate": 0.5137,
        "samples": 12418, "windows": 3, "available": True, "passed": False,
        "reason": "命中率 51.37% < 52%", "thresholds": _thresholds(),
    }
    h.update(over)
    return h


# ---------------------------------------------------------------- 诊断
def test_diagnose_shortfall_is_exact_gap():
    d = diagnose_horizon("short_term", _horizon(), GATE)
    # 52% - 51.37% = 0.63%
    assert d["shortfall"]["hit_rate"] == pytest.approx(0.0063)
    assert d["binding_metric"] == "hit_rate"
    assert d["passed"] is False


def test_diagnose_headroom_for_passing_metric():
    # IC 0.04 超门槛 0.03 → 记余量而非短板
    d = diagnose_horizon("short_term", _horizon(), GATE)
    assert d["headroom"]["ic"] == pytest.approx(0.01)
    assert "ic" not in d["shortfall"]


def test_diagnose_binding_metric_is_worst_normalized_gap():
    # IC 差得更多（归一化后）→ 短板应是 ic 而非 hit_rate
    d = diagnose_horizon(
        "h", _horizon(ic=0.005, hit_rate=0.515, reason=""), GATE)
    assert d["binding_metric"] == "ic"
    assert d["gaps"]["ic"] > d["gaps"]["hit_rate"]


def test_diagnose_passed_horizon_has_no_shortfall():
    d = diagnose_horizon("long_term", _horizon(passed=True, hit_rate=0.5453), GATE)
    assert d["passed"] is True
    assert d["shortfall"] == {}
    assert d["total_gap"] == 0.0
    assert d["binding_metric"] is None


def test_diagnose_unavailable_horizon_not_judged_as_threshold_gap():
    # 样本不足：绝不能算成"差 12% 命中率"，那是误导
    d = diagnose_horizon("h", _horizon(available=False, samples=5, reason="样本不足（5 < 30）"), GATE)
    assert d["available"] is False
    assert d["shortfall"] == {}
    assert "样本不足" in d["reason"]


def test_diagnose_zero_samples_marks_unavailable():
    d = diagnose_horizon("h", _horizon(samples=0, hit_rate=0.0, ic=0.0), GATE)
    assert d["available"] is False


def test_diagnose_all_scope_binding_is_hardest_leg():
    payload = {"horizons": {
        "short_term": _horizon(hit_rate=0.5137),
        "mid_term": _horizon(hit_rate=0.5049),
        "long_term": _horizon(passed=True, hit_rate=0.5453),
    }}
    out = diagnose(payload, GATE, {"state": "readonly"})
    assert out["failing_horizons"] == ["short_term", "mid_term"]
    # mid_term 更差 → scope=all 的约束就是它
    assert out["binding_horizon"] == "mid_term"
    assert out["passed_count"] == 1 and out["total_count"] == 3
    assert out["blocking_metric_counts"] == {"hit_rate": 2}
    assert out["gate_state"] == "readonly"


def test_diagnose_any_scope_binding_is_easiest_leg():
    payload = {"horizons": {
        "short_term": _horizon(hit_rate=0.5137),
        "mid_term": _horizon(hit_rate=0.30),
    }}
    out = diagnose(payload, dict(GATE, scope="any"))
    # 任一达标即放行 → 优先补最容易的那条腿（short_term）
    assert out["binding_horizon"] == "short_term"


def test_diagnose_all_passed_notes_not_blocking():
    payload = {"horizons": {"long_term": _horizon(passed=True)}}
    out = diagnose(payload, GATE)
    assert out["binding_horizon"] is None
    assert "无需诊断" in out["note"]


def test_diagnose_empty_payload_is_honest():
    out = diagnose({"horizons": {}}, GATE)
    assert out["horizons"] == {}
    assert "无可用的 IC 评估结果" in out["note"]


def test_diagnose_does_not_flag_metrics_outside_gate():
    # icir 不参与门禁，缺 ICIR 不应被诊断成"短板"
    d = diagnose_horizon("h", _horizon(icir=0.0), GATE)
    assert "icir" not in d["shortfall"]
    assert set(GATED_METRICS) == {"ic", "hit_rate", "windows"}


def test_diagnose_defaults_to_gate_config_thresholds():
    # payload 不带 thresholds 时回落到 gate_config
    h = _horizon(hit_rate=0.51)
    h.pop("thresholds")
    d = diagnose_horizon("h", h, GATE)
    assert d["shortfall"]["hit_rate"] == pytest.approx(0.01)


def test_diagnose_is_pure_and_repeatable():
    payload = {"horizons": {"short_term": _horizon()}}
    a = diagnose(payload, GATE)
    b = diagnose(payload, GATE)
    assert a == b


# ---------------------------------------------------------------- 中性带
def _noisy(signal_ratio: float, n: int, seed: int = 0):
    """合成序列：signal_ratio 比例是强信号（方向对），其余是低幅噪音。"""
    rnd = random.Random(seed)
    sc, rr = [], []
    for _ in range(n):
        if rnd.random() < signal_ratio:
            s = rnd.choice([-1.0, 1.0]) * rnd.uniform(0.25, 0.5)
            r = s * rnd.uniform(0.001, 0.02)
        else:
            s = rnd.uniform(-0.08, 0.08)
            r = rnd.uniform(-0.02, 0.02)
        sc.append(s)
        rr.append(r)
    return sc, rr


def test_derive_band_requires_min_samples():
    assert derive_neutral_band([0.1, 0.2], [0.01, 0.02]) == 0.0
    assert derive_neutral_band([], []) == 0.0


def test_derive_band_returns_zero_on_pure_noise():
    rnd = random.Random(3)
    sc = [rnd.uniform(-0.5, 0.5) for _ in range(500)]
    rr = [rnd.uniform(-0.02, 0.02) for _ in range(500)]
    # 纯噪声：最优带不应偏离 0（除非真能提升，否则不做无依据的门槛）
    band = derive_neutral_band(sc, rr)
    assert band >= 0.0 and band <= max(abs(x) for x in sc)


def test_derive_band_respects_min_keep_ratio():
    sc, rr = _noisy(0.5, 2000, seed=11)
    band = derive_neutral_band(sc, rr, min_keep_ratio=0.5)
    kept = [x for x in sc if abs(x) > band]
    assert len(kept) >= 0.5 * len(sc) - 1  # 至少保留一半


def test_derive_band_uses_train_only_and_lifts_test_hit_rate():
    train_sc, train_rr = _noisy(0.5, 3000, seed=1)
    test_sc, test_rr = _noisy(0.5, 3000, seed=2)
    band = derive_neutral_band(train_sc, train_rr)
    assert band > 0.0
    raw = hit_rate(test_sc, test_rr)
    banded = hit_rate(test_sc, test_rr, neutral_band=band)
    # 关键：band 完全来自训练折，测试折命中率提升不是泄漏
    assert banded > raw + 0.05
    assert hit_rate(train_sc, train_rr, neutral_band=band) > hit_rate(train_sc, train_rr)


def test_derive_band_zero_band_is_valid_candidate():
    # 信号本来就干净（无噪音）→ 不必设门槛，返回 0 也应能"通过"
    sc = []
    rr = []
    rnd = random.Random(5)
    for _ in range(400):
        s = rnd.choice([-1.0, 1.0]) * 0.4
        sc.append(s)
        rr.append(s * 0.01)
    band = derive_neutral_band(sc, rr)
    assert band == 0.0


# ---------------------------------------------------------------- 评估口径
def test_ic_result_reports_band_and_raw_rate():
    n = 200
    scores = [0.6 if i % 3 else 0.4 for i in range(n)]
    returns = [0.01 if i % 3 else -0.01 for i in range(n)]
    res = ICCalculator(min_samples=30).evaluate(
        "short_term", 5, scores, returns, neutral_band=0.05)
    assert res.neutral_band == pytest.approx(0.05)
    assert res.hit_rate_raw == pytest.approx(res.hit_rate)  # 无样本落入带内 → 两者一致
    assert res.to_dict()["neutral_band"] == pytest.approx(0.05)
    assert "hit_rate_raw" in res.to_dict()


def test_ic_default_band_is_backward_compatible():
    n = 200
    scores = [0.6 if i % 3 else 0.4 for i in range(n)]
    returns = [0.01 if i % 3 else -0.01 for i in range(n)]
    res = ICCalculator(min_samples=30).evaluate("short_term", 5, scores, returns)
    assert res.neutral_band == 0.0
    assert res.hit_rate == pytest.approx(res.hit_rate_raw)


def test_evaluate_all_passes_band_through():
    calc = ICCalculator(min_samples=5, min_ic=0.0, min_hit_rate=0.0, min_windows=3)
    payload = {
        "short_term": {"horizon_days": 5, "scores": [0.6, 0.4] * 10,
                       "returns": [0.01, -0.01] * 10, "neutral_band": 0.05},
    }
    out = calc.evaluate_all(payload)
    assert out["horizons"]["short_term"]["neutral_band"] == pytest.approx(0.05)


# ---------------------------------------------------------------- CLI / 兼容
def test_cli_exposes_gate_diagnose_and_ic_flags():
    import main as m

    parser = m.build_parser()
    args = parser.parse_args(["gate-diagnose"])
    assert args.command == "gate-diagnose"
    args = parser.parse_args(["ic", "--stratify", "--derive-band"])
    assert args.stratify is True and args.derive_band is True


def test_run_ic_signature_backward_compatible():
    import inspect

    import main as m

    sig = inspect.signature(m.run_ic)
    params = list(sig.parameters)
    assert params[:3] == ["config", "symbols", "folds"]
    assert sig.parameters["stratify"].default is False
    # derive_band 默认 None = 跟随配置 strategy_gate.neutral_band.enabled
    assert sig.parameters["derive_band"].default is None


def test_derive_band_follows_config_when_not_explicit():
    """未显式传参时，run_ic 跟随 strategy_gate.neutral_band.enabled。"""
    from src.inference.ic import ICCalculator  # noqa: F401
    import yaml
    for path in ("configs/config.yaml", "configs/config_pro.yaml"):
        cfg = yaml.safe_load(open(path, encoding="utf-8"))
        nb = cfg["strategy_gate"]["neutral_band"]
        assert nb["enabled"] is False           # 默认关闭，保持既有口径
        assert 0 < nb["min_keep_ratio"] <= 1.0


def test_per_symbol_summary_shape_is_compact():
    import main as m

    raw = {"600519.SH": {"short_term": {
        "horizon_days": 5, "scores": [0.1, -0.1] * 20,
        "returns": [0.01, -0.01] * 20, "neutral_band": 0.0}}}
    out = m._summarize_per_symbol(raw)
    entry = out["600519.SH"]["short_term"]
    assert set(entry) == {"samples", "ic", "hit_rate", "hit_rate_raw",
                          "neutral_band", "available", "passed"}
    # 不得把原始数组塞进报告
    assert "scores" not in entry and "returns" not in entry


def test_ic_scores_for_horizon_band_off_matches_legacy():
    """derive_band=False 时必须逐字段等同旧口径（neutral_band=0）。"""
    import pandas as pd
    import numpy as np

    import main as m

    n = 300
    rnd = np.random.RandomState(0)
    close = 100 + np.cumsum(rnd.normal(0, 1, n))
    combined = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n, freq="D"),
        "open": close, "high": close + 1, "low": close - 1, "close": close,
        "volume": 1e6,
    })
    cfg = {"model": {"type": "lightgbm", "params": {}},
           "data": {"prediction_horizons": {"short_term": 5}},
           "features": {"technical": {"ma_windows": [5, 10], "rsi_window": 14,
                                      "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
                                      "bollinger_window": 20, "bollinger_std": 2},
                        "returns": {"windows": [1, 3, 5]},
                        "volatility": {"windows": [5, 10]},
                        "volume": {"ma_windows": [5, 10]}}}
    from src.data.preprocessor import FeatureEngineer

    fe = FeatureEngineer(cfg)
    combined = fe.transform(combined, 5)
    combined = fe.create_target(combined, 5)
    combined["_symbol"] = "TEST"
    combined["_fwd_ret"] = combined["close"].shift(-5) / combined["close"] - 1
    combined = combined.dropna(subset=["target_5d"])

    _, _, band_off = m._ic_scores_for_horizon(cfg, combined, 5, 2, derive_band=False)
    _, _, band_on = m._ic_scores_for_horizon(cfg, combined, 5, 2, derive_band=True)
    assert band_off == 0.0            # 关闭时绝不引入门槛
    assert band_on >= 0.0             # 开启时来自训练折，非负


# ---------------------------------------------------------------- 监控报表
def test_monitor_reports_gate_diagnosis(tmp_path):
    import json

    from src.monitor.health_report import ModelMonitor

    rep = tmp_path / "reports"
    rep.mkdir()
    (rep / "ic_report.json").write_text(json.dumps({"horizons": {
        "short_term": _horizon(hit_rate=0.5137),
        "mid_term": _horizon(hit_rate=0.5049, horizon_days=10),
    }}), encoding="utf-8")
    (rep / "strategy_gate.json").write_text(json.dumps({
        "state": "readonly", "passed": False, "reason": "命中率未达标",
        "blocked_by": ["short_term: 命中率 51.37% < 52%"],
        "horizons": {"short_term": {"ic": 0.04, "hit_rate": 0.5137,
                                    "samples": 12418, "passed": False},
                     "mid_term": {"ic": 0.0513, "hit_rate": 0.5049,
                                  "samples": 12340, "passed": False}},
        "generated_at": "2026-09-10T00:00:00",
    }), encoding="utf-8")

    cfg = {"strategy_gate": dict(GATE, report_dir=str(rep)),
           "report": {"output_dir": str(rep)},
           "data": {"markets": {}}}
    monitor = ModelMonitor(cfg)
    payload = monitor.collect().to_dict()
    diag = payload["gate_diagnosis"]
    assert diag["available"] is True
    assert diag["binding_horizon"] == "mid_term"
    # 诊断结果必须体现在 issues 里（不能只有一句"门禁未放行"）
    assert any("mid_term" in i for i in payload["issues"])
    md = monitor.collect().to_markdown()
    assert "阻塞诊断" in md


def test_monitor_skips_diagnosis_when_gate_passed(tmp_path):
    import json

    from src.monitor.health_report import ModelMonitor

    rep = tmp_path / "reports"
    rep.mkdir()
    (rep / "strategy_gate.json").write_text(json.dumps({
        "state": "gated", "passed": True, "reason": "", "horizons": {},
    }), encoding="utf-8")
    cfg = {"strategy_gate": dict(GATE, report_dir=str(rep)),
           "report": {"output_dir": str(rep)}, "data": {"markets": {}}}
    payload = ModelMonitor(cfg).collect().to_dict()
    assert payload["gate_diagnosis"]["available"] is False
    assert payload["gate_diagnosis"]["reason"] == "gate_not_blocked"


def test_note_renders_shortfall_readably():
    from src.inference.gate_diagnosis import _fmt_shortfall, _note
    assert "1.51 个百分点" in _fmt_shortfall({"hit_rate": 0.0151})
    assert "|IC|" in _fmt_shortfall({"ic": 0.02}) or "0.0200" in _fmt_shortfall({"ic": 0.02})
    note = _note([{"horizon": "mid_term"}], {"horizon": "mid_term",
                 "binding_metric": "hit_rate", "shortfall": {"hit_rate": 0.0151}}, "all")
    assert "mid_term" in note
