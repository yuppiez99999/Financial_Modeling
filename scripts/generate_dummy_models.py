#!/usr/bin/env python3
"""生成占位 LightGBM 模型文件供最小化测试使用"""
from pathlib import Path
import joblib

from src.utils.dummy_models import DummyModel, DummyScaler

MODELS = [
    ("lightgbm_short_term_5d.pkl", 3),
    ("lightgbm_mid_term_10d.pkl", 3),
    ("lightgbm_long_term_20d.pkl", 3),
]


def main():
    root = Path(__file__).resolve().parents[1]
    save_dir = root / "models"
    save_dir.mkdir(parents=True, exist_ok=True)

    for fname, nfeat in MODELS:
        path = save_dir / fname
        data = {"model": DummyModel(n_features=nfeat), "scaler": DummyScaler()}
        joblib.dump(data, path)
        print(f"Wrote {path}")

if __name__ == '__main__':
    main()
