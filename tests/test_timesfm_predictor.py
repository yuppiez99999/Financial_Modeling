"""测试 TimesFMFinancePredictor 占位回退逻辑。"""
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.timesfm_predictor import TimesFMFinancePredictor  # noqa: E402


def test_predict_classification_placeholder():
    pred = TimesFMFinancePredictor(horizon_days=5, context_days=10)
    series = np.random.default_rng(0).normal(100, 1, 100)
    res = pred.predict_classification(series, horizon=5, symbol="AAA")
    assert res["backend"] == "placeholder"
    assert res["prediction"] in (0, 1)
    assert 0 <= res["probability"] <= 1
    assert "direction" in res


def test_predict_classification_short_series():
    pred = TimesFMFinancePredictor()
    res = pred.predict_classification(np.array([1.0]), horizon=5)
    assert res.get("error")
    assert res["prediction"] == 0


def test_forecast_placeholder():
    pred = TimesFMFinancePredictor()
    fc = pred.forecast(np.random.default_rng(1).normal(100, 1, 50), horizon=7)
    assert fc.shape == (7,)


def test_to_series_handles_types():
    pred = TimesFMFinancePredictor()
    assert pred._to_series([1, 2, 3]).shape == (3,)
    assert pred._to_series(np.array([[1.0], [2.0]])).shape == (2,)
    import pandas as pd
    assert pred._to_series(pd.Series([1, 2, 3])).shape == (3,)
