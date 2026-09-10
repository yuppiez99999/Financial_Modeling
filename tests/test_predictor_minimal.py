import sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.inference.predictor import PredictionEngine


class DummyCollector:
    def __init__(self, cfg):
        pass

    def load_cached(self, symbol):
        import pandas as pd
        import numpy as np

        dates = pd.date_range(end=pd.Timestamp.today(), periods=100)
        closes = 100 + np.cumsum(np.random.randn(100)) * 0.5
        df = pd.DataFrame({"date": dates, "close": closes})
        return df

    def _fetch_with_fallback(self, symbol):
        return self.load_cached(symbol)


class DummyFeatureEngineer:
    def __init__(self, cfg):
        pass

    def transform(self, df, horizon_days):
        out = df.copy()
        out["f1"] = out["close"].pct_change().fillna(0)
        out["f2"] = out["close"].rolling(3).mean().bfill()
        out["f3"] = out["close"].rolling(5).std().fillna(0)
        return out

    def get_feature_columns(self, df_features, horizon_days):
        return ["f1", "f2", "f3"]


class DummyTFM:
    def __init__(self, horizon_days=5, context_days=252, verbose=False):
        self.horizon_days = horizon_days

    def predict_classification(self, close_prices, horizon, symbol=None):
        return {"prediction": 1, "probability": 0.8}


@pytest.fixture()
def cfg():
    cfg_path = PROJECT_ROOT / "configs" / "config.yaml"
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    return cfg


def apply_monkeypatches(monkeypatch):
    import src.inference.predictor as predmod

    predmod.DataCollector = DummyCollector
    predmod.FeatureEngineer = DummyFeatureEngineer
    # ensure TimesFMFinancePredictor uses a safe placeholder by default in tests
    import src.timesfm_predictor as tfm_mod
    tfm_mod.TimesFMFinancePredictor = DummyTFM


def test_lightgbm_path(cfg, monkeypatch):
    apply_monkeypatches(monkeypatch)
    cfg["model"]["type"] = "lightgbm"
    engine = PredictionEngine(cfg)
    engine.load_models("lightgbm")
    res = engine.predict("TEST", "short_term")
    assert isinstance(res, dict)
    assert "prediction" in res or "error" in res


def test_timesfm_path(cfg, monkeypatch):
    apply_monkeypatches(monkeypatch)
    # monkeypatch TimesFM predictor to avoid external dependency
    import src.timesfm_predictor as tfm_mod

    monkeypatch.setattr(tfm_mod, "TimesFMFinancePredictor", DummyTFM)

    cfg["model"]["type"] = "timesfm"
    engine = PredictionEngine(cfg)
    engine.load_models("timesfm")
    res = engine.predict("TEST", "short_term")
    assert isinstance(res, dict)
    assert res.get("prediction") in (0, 1)


def test_ensemble_path(cfg, monkeypatch):
    apply_monkeypatches(monkeypatch)
    # ensure TimesFM is monkeypatched
    import src.timesfm_predictor as tfm_mod

    monkeypatch.setattr(tfm_mod, "TimesFMFinancePredictor", DummyTFM)

    cfg["model"]["type"] = "ensemble"
    engine = PredictionEngine(cfg)
    engine.load_models("ensemble")
    res = engine.predict("TEST", "short_term")
    assert isinstance(res, dict)
    assert "prediction" in res or "error" in res
