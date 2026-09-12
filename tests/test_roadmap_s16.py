"""S16 / H1 路线测试：保形预测覆盖率校准 + 口径对照 + 多时段复验（T16.1~T16.3）。

设计要点（与 S11~S15 测试同构）：
  - 全部离线（合成数据 + tmp 目录），CI 不触网、不重训真实模型；
  - 边界优先：不改门禁（affects_gate 恒 false）、不自动定 α / 阈值；
  - 不猜：校准样本不足 → available=false；名义覆盖不可达 → 不外推；
  - 无前视：覆盖率只在**更晚的复验段**读数，分段不重叠。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.calibration_ab import (  # noqa: E402
    compare_confidence_sources,
    rolling_holdout_verify,
    split_time_segments,
)
from src.eval.conformal import (  # noqa: E402
    build_calibration_report,
    conformal_quantile,
    empirical_coverage,
    fit_conformal_interval,
    mean_interval_width,
    save_calibration_report,
)


def _data(n=600, seed=7):
    rng = np.random.default_rng(seed)
    y = rng.normal(0.0, 0.02, n)
    p = 0.5 * y + rng.normal(0, 0.01, n)   # 预测与真实同向但含噪声
    return y, p


# ----------------------------------------------------------------------
# T16.1 保形区间与覆盖率
# ----------------------------------------------------------------------
class TestConformalInterval:
    def test_quantile_finite_sample_correction(self):
        """分位数必须用 (n+1)(1-α)/n 保守水平，不能直接用 1-α。"""
        r = np.linspace(0, 1, 100)
        q = conformal_quantile(r, 0.1, n_cal=100)
        assert q == pytest.approx(np.quantile(r, 0.91), abs=1e-9)

    def test_unreachable_nominal_coverage_returns_none_not_inf(self):
        """校准样本太少导致名义覆盖不可达 → None，不用 inf 冒充。"""
        assert conformal_quantile([0.1, 0.2], 0.01, n_cal=2) is None

    def test_insufficient_calibration_not_guessed(self):
        iv = fit_conformal_interval([1.0] * 5, [0.0] * 5, 0.1)
        assert iv["available"] is False and iv["q"] is None
        assert "校准样本不足" in iv["reason"]

    def test_interval_covers_expected_share(self):
        y, p = _data()
        iv = fit_conformal_interval(y[:200], p[:200], 0.2, y_pred_use=p[200:])
        cov = empirical_coverage(y[200:], iv["lower"], iv["upper"])
        assert cov is not None and cov >= 0.7  # 名义 80%，宽松下界

    def test_width_positive_and_none_on_empty(self):
        assert mean_interval_width([], []) is None
        assert mean_interval_width([1.0, 2.0], [3.0, 4.0]) == pytest.approx(2.0)


class TestCalibrationReport:
    def test_report_shape_and_boundaries(self):
        y, p = _data()
        rep = build_calibration_report(y, p, n_train=300, n_cal=150, horizon_days=5)
        assert rep["kind"] == "conformal_calibration"
        assert rep["affects_gate"] is False
        assert rep["segments"]["n_verify"] == 150
        assert rep["available"] is True
        assert rep["backend"] in ("mapie", "native")

    def test_coverage_decreases_as_alpha_increases(self):
        """α 越大区间越窄 → 经验覆盖率必须单调不增（口径自洽性）。"""
        y, p = _data(seed=11)
        rep = build_calibration_report(y, p, n_train=300, n_cal=150, horizon_days=5)
        covs = [r["empirical_coverage"] for r in rep["rows"] if r["available"]]
        assert len(covs) >= 3
        for a, b in zip(covs, covs[1:]):
            assert b <= a + 0.05  # 允许抽样噪声，但不得系统性反向

    def test_insufficient_segments_reported_not_fabricated(self):
        y, p = _data(n=60)
        rep = build_calibration_report(y, p, n_train=30, n_cal=20, horizon_days=5)
        assert rep["available"] is False
        assert "样本不足" in rep["reason"] and rep["rows"] == []

    def test_save_report_to_calibration_dir(self, tmp_path):
        y, p = _data()
        rep = build_calibration_report(y, p, n_train=300, n_cal=150, horizon_days=10)
        path = save_calibration_report(rep, out_dir=str(tmp_path / "cal"))
        assert path.name == "conformal_10d.json" and path.exists()


# ----------------------------------------------------------------------
# T16.2 口径对照
# ----------------------------------------------------------------------
class TestConfidenceSourceAB:
    def test_two_arms_reported_and_no_gate_change(self):
        y, p = _data(seed=3)
        lo, hi, mid = p - 0.03, p + 0.03, np.full(p.size, 1.0)
        ab = compare_confidence_sources(p, y, lower=lo, upper=hi, mid=mid)
        assert ab["kind"] == "confidence_source_ab"
        assert ab["affects_gate"] is False
        assert ab["interval_provided"] is True
        assert all(r["proba_distance"]["available"] for r in ab["rows"])
        assert any(r["interval_width"]["available"] for r in ab["rows"])

    def test_missing_interval_not_fabricated(self):
        y, p = _data(seed=4)
        ab = compare_confidence_sources(p, y)
        assert ab["interval_provided"] is False
        assert ab["verdict"] == "unavailable"
        assert "区间口径不可用" in ab["verdict_detail"]

    def test_verdict_is_one_of_known_states(self):
        y, p = _data(seed=5)
        ab = compare_confidence_sources(p, y, lower=p - 0.05, upper=p + 0.05,
                                        mid=np.ones_like(p))
        assert ab["verdict"] in ("interval_more_separative",
                                 "proba_distance_more_separative",
                                 "no_material_difference", "unavailable")

    def test_empty_input_available_false(self):
        ab = compare_confidence_sources([], [])
        assert ab["available"] is False and ab["reason"] == "无样本"


# ----------------------------------------------------------------------
# T16.3 多时段滚动保留期复验
# ----------------------------------------------------------------------
class TestRollingHoldout:
    def test_segments_are_contiguous_and_non_overlapping(self):
        segs = split_time_segments(1000, segments=4, min_samples=100)
        assert len(segs) == 4
        for a, b in zip(segs, segs[1:]):
            assert a["end"] == b["start"]
        assert sum(s["samples"] for s in segs) == 1000

    def test_insufficient_samples_returns_empty_not_guessed(self):
        assert split_time_segments(150, segments=4, min_samples=100) == []

    def test_rolling_report_reason_when_insufficient(self):
        p = np.linspace(0, 1, 120)
        rv = rolling_holdout_verify(p, p, segments=4, min_samples=100)
        assert rv["available"] is False
        assert "样本不足" in rv["reason"] and rv["segments"] == []

    def test_monotone_signal_detected_as_stable(self):
        """构造与阈值同向的合成数据：各段命中率随 thr 单调不减 → stable。"""
        rng = np.random.default_rng(0)
        n = 2000
        p = np.clip(0.5 + 0.45 * (rng.random(n) - 0.5) * 2, 0.01, 0.99)
        r = (p - 0.5) * 0.2 + rng.normal(0, 0.05, n)
        rv = rolling_holdout_verify(p, r, segments=4, min_samples=100)
        assert rv["available"] is True
        assert rv["segments_used"] == 4
        assert rv["stability"] in ("stable", "partially_stable")
        assert rv["affects_gate"] is False

    def test_noise_not_declared_stable_as_evidence(self):
        """纯噪声不得被包装成达标证据：结论只能是稳定性描述，且标注不作证据。"""
        rng = np.random.default_rng(1)
        n = 1600
        p = rng.random(n)
        r = rng.normal(0, 1, n)
        rv = rolling_holdout_verify(p, r, segments=4, min_samples=100)
        assert rv["stability"] in ("stable", "partially_stable", "unstable")
        assert "不作达标证据" in rv["note"]
