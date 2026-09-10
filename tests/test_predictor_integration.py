import sys
from pathlib import Path
import yaml

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
        out["f2"] = out["close"].rolling(3).mean().ffill()
        out["f3"] = out["close"].rolling(5).std().fillna(0)
        return out

    def get_feature_columns(self, df_features, horizon_days):
        return ["f1", "f2", "f3"]


def test_ensemble_with_placeholders(tmp_path, monkeypatch):
    # 加载配置
    cfg_path = PROJECT_ROOT / "configs" / "config.yaml"
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))

    # 让引擎使用测试替身的数据管道
    import src.inference.predictor as predmod

    predmod.DataCollector = DummyCollector
    predmod.FeatureEngineer = DummyFeatureEngineer

    cfg["model"]["type"] = "ensemble"
    engine = PredictionEngine(cfg)
    # 加载 ensemble（会优先使用 models/ 下的占位 pickles）
    engine.load_models("ensemble")

    res = engine.predict("TEST", "short_term")
    assert isinstance(res, dict)
    assert "prediction" in res or "error" in res
