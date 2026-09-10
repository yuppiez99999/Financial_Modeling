"""将已存在的占位 pickle 包装为包含 'scaler' 的 dict 结构。

当 models/ 下已有旧的占位 pickle（没有 scaler 字段）时，本脚本会将其
包装为 {'model': obj, 'scaler': DummyScaler()} 并重新落盘。
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.dummy_models import DummyScaler  # noqa: E402

p = PROJECT_ROOT / "models"
filenames = [
    "timesfm_short_term_5d.pkl",
    "timesfm_mid_term_10d.pkl",
    "timesfm_long_term_20d.pkl",
]
for fn in filenames:
    path = p / fn
    if not path.exists():
        print("missing", path)
        continue
    obj = joblib.load(path)
    # 如果已是 dict 且包含 model，则跳过
    if isinstance(obj, dict) and "model" in obj:
        print("already wrapped", path)
        continue
    wrapped = {"model": obj, "scaler": DummyScaler()}
    joblib.dump(wrapped, path)
    print("rewrote", path)
