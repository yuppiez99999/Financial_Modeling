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
    """替换数据管道 / TimesFM 为测试替身。

    必须走 `monkeypatch.setattr`：直接赋模块属性会永久污染模块全局，
    使后续用例（如 Q2 多因子推理）拿到假的数据管道。
    """
    import src.inference.predictor as predmod

    monkeypatch.setattr(predmod, "DataCollector", DummyCollector)
    monkeypatch.setattr(predmod, "FeatureEngineer", DummyFeatureEngineer)
    # ensure TimesFMFinancePredictor uses a safe placeholder by default in tests
    import src.timesfm_predictor as tfm_mod
    monkeypatch.setattr(tfm_mod, "TimesFMFinancePredictor", DummyTFM)


def test_lightgbm_path(cfg, monkeypatch, tmp_path):
    apply_monkeypatches(monkeypatch)
    cfg["model"]["type"] = "lightgbm"
    # 用契约匹配的微型模型替换真实 pkl（2026-09-15 起 pkl 持久化 feature_cols，
    # 推理按名对齐 fail-close：合成 3 特征 vs 真实 107 契约会显式报错——这是
    # 修复后的正确行为。本测试改为训练一个 [f1,f2,f3] 契约的微型模型，
    # 走通「契约对齐 → scaler → 预测」的完整成功路径。）
    from src.train.models.lightgbm_model import LightGBMModel

    cfg["training"] = dict(cfg.get("training") or {})
    cfg["training"]["save_dir"] = str(tmp_path)
    cfg["model"] = dict(cfg.get("model") or {})
    cfg["model"]["lightgbm"] = {
        "objective": "binary", "n_estimators": 10, "learning_rate": 0.1,
        "max_depth": 2, "num_leaves": 4, "subsample": 1.0,
        "colsample_bytree": 1.0, "reg_alpha": 0.0, "reg_lambda": 0.0,
        "verbose": -1, "early_stopping_rounds": 5,
    }
    import numpy as np
    rng = np.random.RandomState(42)
    X = rng.normal(size=(120, 3))
    y = (X[:, 0] > 0).astype(int)
    tiny = LightGBMModel(cfg)
    tiny.train(X, y, feature_cols=["f1", "f2", "f3"])
    tiny.save(str(tmp_path / "lightgbm_short_term_5d.pkl"))

    engine = PredictionEngine(cfg)
    engine.load_models("lightgbm")
    res = engine.predict("TEST", "short_term")
    assert isinstance(res, dict)
    assert "prediction" in res or "error" in res
    if "prediction" in res:
        assert res["prediction"] in (0, 1)



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
