"""测试占位模型与缩放器。"""
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.dummy_models import DummyModel, DummyScaler  # noqa: E402


def test_dummy_scaler_is_identity():
    scaler = DummyScaler()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(scaler.transform(X), X)
    np.testing.assert_allclose(scaler.fit_transform(X), X)
    np.testing.assert_allclose(scaler.inverse_transform(X), X)
    assert scaler.fit(X) is scaler


def test_dummy_model_predict_and_proba():
    model = DummyModel()
    X = np.zeros((3, 5))
    preds = model.predict(X)
    assert preds.shape == (3,)
    assert set(np.unique(preds)) <= {0, 1}
    proba = model.predict_proba(X)
    assert proba.shape == (3, 2)
    np.testing.assert_allclose(proba, 0.5)


def test_dummy_model_predict_classification():
    model = DummyModel()
    res = model.predict_classification(np.array([1, 2, 3]), horizon=5, symbol="T")
    assert "prediction" in res
    assert res["symbol"] == "T"
    assert res["horizon"] == 5
