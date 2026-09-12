"""S19 / H4 路线测试：概率校准层（T19.1~T19.3）。

设计要点：
  - 全部离线（合成概率 + tmp 目录），CI 不触网；
  - 边界优先：不改门禁、不改现行阈值、不自动选校准器；
  - 无前视：校准器只在**校准段**拟合，指标只在**更晚的复验段**读数；
  - 不猜：样本不足 / 参数缺失 → 不拟合、不校准、如实标注。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.probability_calibration import (  # noqa: E402
    METHOD_ISOTONIC,
    METHOD_PLATT,
    brier_score,
    calibrate_and_evaluate,
    expected_calibration_error,
    fit_calibrator,
    reliability_curve,
    threshold_curve_after_calibration,
)
from src.inference.probability_calibrator import (  # noqa: E402
    calibrate_probability,
    enrich_prediction,
    load_calibration,
    uncertainty_from_probability,
)


def _overconfident(n=1500, seed=0):
    """构造"过度自信"的模型概率：真实频率低于自报概率。"""
    rng = np.random.default_rng(seed)
    true_p = rng.random(n)
    y = (rng.random(n) < true_p).astype(int)
    p = np.clip(0.5 + (true_p - 0.5) * 2.2, 0.01, 0.99)
    return p, y


# ----------------------------------------------------------------------
# T19.1 校准误差
# ----------------------------------------------------------------------
class TestCalibrationMetrics:
    def test_brier_perfect_prediction_is_zero(self):
        assert brier_score([1.0, 0.0], [1, 0]) == pytest.approx(0.0)

    def test_brier_none_on_empty(self):
        assert brier_score([], []) is None

    def test_ece_zero_when_perfectly_calibrated(self):
        """每个概率箱里真实频率 == 预测均值 → ECE 严格为 0。"""
        p = np.array([0.2] * 500 + [0.8] * 500)
        # 0.2 箱：真实频率 0.2（100 正 / 400 负）；0.8 箱：真实频率 0.8
        y = np.array([1] * 100 + [0] * 400 + [1] * 400 + [0] * 100)
        assert expected_calibration_error(p, y) == pytest.approx(0.0)

    def test_reliability_empty_bin_not_filled(self):
        rows = reliability_curve([0.95, 0.96], [1, 1], bins=10)
        assert sum(1 for r in rows if not r["available"]) >= 5
        for r in rows:
            if not r["available"]:
                assert r["mean_predicted"] is None and r["observed_frequency"] is None

    def test_fit_insufficient_not_fitted(self):
        r = fit_calibrator([0.5] * 10, [1] * 10, method=METHOD_ISOTONIC)
        assert r["available"] is False and "样本不足" in r["reason"]

    def test_fit_single_class_not_fitted(self):
        r = fit_calibrator(np.linspace(0.1, 0.9, 200).tolist(), [1] * 200,
                           method=METHOD_ISOTONIC)
        assert r["available"] is False

    def test_unknown_method_rejected(self):
        r = fit_calibrator(np.linspace(0.1, 0.9, 200).tolist(),
                           ([0] * 100 + [1] * 100), method="magic")
        assert r["available"] is False and "未知校准方法" in r["reason"]


class TestCalibrateAndEvaluate:
    def test_segments_strictly_ordered_and_boundaries(self):
        p, y = _overconfident()
        rep = calibrate_and_evaluate(p, y, n_train=800, n_cal=400, horizon_name="5d")
        seg = rep["segments"]
        assert seg["n_train"] == 800 and seg["n_calibration"] == 400
        assert seg["n_verify"] == seg["n_total"] - 1200
        assert rep["affects_gate"] is False

    def test_calibration_reduces_ece_on_holdout(self):
        """过度自信的概率经校准后，复验段 ECE 应显著下降。"""
        p, y = _overconfident()
        rep = calibrate_and_evaluate(p, y, n_train=800, n_cal=400)
        assert rep["available"] is True
        assert rep["methods"][METHOD_ISOTONIC]["ece"] < rep["baseline"]["ece"]
        assert rep["methods"][METHOD_ISOTONIC]["improved"] is True

    def test_metrics_read_on_verify_not_calibration_segment(self):
        """Brier 必须来自复验段：篡改校准段之后的样本会改变读数。"""
        p, y = _overconfident(seed=5)
        rep1 = calibrate_and_evaluate(p, y, n_train=800, n_cal=400)
        flipped = y.copy()
        flipped[-100:] = 1 - flipped[-100:]
        rep2 = calibrate_and_evaluate(p, flipped, n_train=800, n_cal=400)
        assert rep1["baseline"]["brier"] != rep2["baseline"]["brier"]

    def test_insufficient_segments_not_guessed(self):
        p, y = _overconfident(n=200)
        rep = calibrate_and_evaluate(p, y, n_train=50, n_cal=20)
        assert rep["available"] is False and "样本不足" in rep["reason"]

    def test_conclusion_does_not_auto_decide(self):
        p, y = _overconfident()
        rep = calibrate_and_evaluate(p, y, n_train=800, n_cal=400)
        assert "T19.4" in rep["conclusion"]
        assert "T19.4" in rep["note"]


class TestThresholdCurveRecalculation:
    def test_curve_compare_does_not_change_threshold(self):
        p, y = _overconfident(seed=2)
        pc = np.clip(p * 0.8 + 0.1, 0.01, 0.99)
        fwd = np.where(y == 1, 0.02, -0.02)
        res = threshold_curve_after_calibration(p, pc, y, fwd)
        assert res["available"] is True
        assert res["affects_gate"] is False
        assert "现行置信度阈值不变" in res["note"]
        assert res["verdict"] in ("calibration_improves_monotonicity",
                                 "calibration_worsens_monotonicity",
                                 "no_material_change", "unavailable")

    def test_empty_not_guessed(self):
        res = threshold_curve_after_calibration([], [], [], [])
        assert res["available"] is False


# ----------------------------------------------------------------------
# T19.3 推理期校准（API 字段）
# ----------------------------------------------------------------------
class TestInferenceCalibrator:
    def test_no_param_file_means_not_applied(self, tmp_path):
        out = enrich_prediction({"probability": 0.7, "confidence": 0.7},
                                "short_term", directory=str(tmp_path))
        assert out["calibration_applied"] is False
        assert out["calibrated_probability"] is None
        assert "main.py calibration" in out["calibration_reason"]

    def test_existing_fields_untouched(self, tmp_path):
        """向后兼容：既有字段逐字段不变（28 侧可渐进消费）。"""
        (tmp_path / "probability_calibration_short_term.json").write_text(
            json.dumps({"method": "platt",
                        "params": {"coef": 2.0, "intercept": -0.5}}), encoding="utf-8")
        pred = {"probability": 0.7, "confidence": 0.7, "direction": "看涨"}
        out = enrich_prediction(pred, "short_term", directory=str(tmp_path))
        assert out["probability"] == 0.7 and out["confidence"] == 0.7
        assert out["direction"] == "看涨"
        assert out["calibrated_probability"] is not None
        assert out["calibration_applied"] is True
        assert out["calibration_method"] == "platt"

    def test_corrupt_param_file_not_guessed(self, tmp_path):
        (tmp_path / "probability_calibration_short_term.json").write_text(
            "{broken", encoding="utf-8")
        assert load_calibration("short_term", directory=str(tmp_path)) is None
        out = enrich_prediction({"probability": 0.6}, "short_term",
                                directory=str(tmp_path))
        assert out["calibration_applied"] is False

    def test_unknown_method_param_rejected(self, tmp_path):
        (tmp_path / "probability_calibration_short_term.json").write_text(
            json.dumps({"method": "platt", "params": {"coef": "abc"}}), encoding="utf-8")
        # 套用失败 → 返回原概率，不抛异常
        assert calibrate_probability(0.7, json.loads(
            (tmp_path / "probability_calibration_short_term.json").read_text())) is not None

    def test_error_prediction_untouched(self, tmp_path):
        pred = {"error": "模型未加载"}
        assert enrich_prediction(pred, "short_term", directory=str(tmp_path)) == pred

    def test_uncertainty_semantics(self):
        assert uncertainty_from_probability(0.5) == pytest.approx(1.0)
        assert uncertainty_from_probability(1.0) == pytest.approx(0.0)
        assert uncertainty_from_probability(0.75) == pytest.approx(0.5)

    def test_isotonic_interpolation_clipped(self):
        cal = {"method": "isotonic",
               "params": {"x_thresholds": [0.0, 0.5, 1.0],
                          "y_thresholds": [0.1, 0.4, 0.9]}}
        assert calibrate_probability(0.6, cal) == pytest.approx(0.5, abs=1e-6)
        assert calibrate_probability(0.0, cal) == pytest.approx(0.1, abs=1e-6)
        assert calibrate_probability(1.0, cal) == pytest.approx(0.9, abs=1e-5)

    def test_isotonic_shape_mismatch_returns_raw(self):
        cal = {"method": "isotonic",
               "params": {"x_thresholds": [0.0, 1.0], "y_thresholds": [0.5]}}
        assert calibrate_probability(0.7, cal) == pytest.approx(0.7)
