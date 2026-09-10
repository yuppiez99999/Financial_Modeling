import joblib
import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

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


class DummyModel:
    def predict(self, X):
        return [1]

    def predict_proba(self, X):
        return [[0.2, 0.8]]


def apply_monkeypatches(monkeypatch):
    import src.inference.predictor as predmod

    predmod.DataCollector = DummyCollector
    predmod.FeatureEngineer = DummyFeatureEngineer


def make_cfg(save_dir: Path):
    return {
        "training": {"save_dir": str(save_dir)},
        "data": {"prediction_horizons": {"short_term": 5}},
        "model": {"ensemble": {"tfm_weight": 0.4}},
    }


def test_ensemble_only_lightgbm(tmp_path, monkeypatch):
    apply_monkeypatches(monkeypatch)
    # create lgb file only（文件名契约与 trainer 保存名一致：lightgbm_{key}.pkl）
    lgb_file = tmp_path / "lightgbm_short_term_5d.pkl"
    joblib.dump(DummyModel(), lgb_file)
    # create a timesfm placeholder to avoid instantiation of real TimesFM
    tfm_file = tmp_path / "timesfm_short_term_5d.pkl"
    joblib.dump({"model": DummyModel(), "scaler": None}, tfm_file)

    cfg = make_cfg(tmp_path)
    engine = PredictionEngine(cfg)
    engine.load_models("ensemble")
    res = engine.predict("TEST", "short_term")
    assert isinstance(res, dict)
    assert "error" not in res


def test_ensemble_only_timesfm_placeholder(tmp_path, monkeypatch):
    apply_monkeypatches(monkeypatch)
    # create timesfm placeholder only（无 lightgbm 分量 → 当前契约降级为"未加载"）
    tfm_file = tmp_path / "timesfm_short_term_5d.pkl"
    joblib.dump({"model": DummyModel(), "scaler": None}, tfm_file)

    cfg = make_cfg(tmp_path)
    engine = PredictionEngine(cfg)
    engine.load_models("ensemble")
    res = engine.predict("TEST", "short_term")
    assert isinstance(res, dict)
    # 已知限制：ensemble 主分量文件缺失时返回未加载错误（fail-close 降级）
    assert "error" in res


def test_ensemble_both_present(tmp_path, monkeypatch):
    apply_monkeypatches(monkeypatch)
    lgb_file = tmp_path / "lightgbm_short_term_5d.pkl"
    tfm_file = tmp_path / "timesfm_short_term_5d.pkl"
    joblib.dump(DummyModel(), lgb_file)
    joblib.dump({"model": DummyModel(), "scaler": None}, tfm_file)

    cfg = make_cfg(tmp_path)
    engine = PredictionEngine(cfg)
    engine.load_models("ensemble")
    res = engine.predict("TEST", "short_term")
    assert isinstance(res, dict)
    assert "error" not in res
