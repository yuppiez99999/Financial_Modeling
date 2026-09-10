#!/usr/bin/env python3
"""最小化预测器验证脚本

目标：在不安装 TimesFM 且无已训练 LightGBM 模型的环境下，验证
`PredictionEngine` 能够优雅处理缺失模型或数据并返回可读错误。
"""
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import PredictionEngine


class DummyCollector:
    def __init__(self, cfg):
        pass

    def load_cached(self, symbol):
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
        out["f2"] = out["close"].rolling(3).mean().fillna(method="bfill")
        out["f3"] = out["close"].rolling(5).std().fillna(0)
        return out

    def get_feature_columns(self, df_features, horizon_days):
        return ["f1", "f2", "f3"]


def main():
    cfg_path = PROJECT_ROOT / "configs" / "config.yaml"
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))

    # 强制使用 lightgbm（本地通常无模型文件）
    cfg["model"]["type"] = "lightgbm"

    # Monkeypatch PredictionEngine 的 DataCollector 和 FeatureEngineer
    import src.inference.predictor as predmod

    predmod.DataCollector = DummyCollector
    predmod.FeatureEngineer = DummyFeatureEngineer

    engine = PredictionEngine(cfg)
    # 尝试加载（若无模型，应该只是警告）
    engine.load_models(cfg["model"]["type"])

    result = engine.predict("TEST", "short_term")
    print("\n=== 最小化测试结果 ===")
    print(result)


if __name__ == "__main__":
    main()
