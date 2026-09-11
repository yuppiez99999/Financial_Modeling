"""生成占位（placeholder）模型 pickle 文件。

用途：在没有 PyTorch / TimesFM 的测试与 CI 环境中，保证
PredictionEngine 可加载占位模型并通过测试。

生成的文件（位于 models/）：
- timesfm_short_term_5d.pkl
- timesfm_mid_term_10d.pkl
- timesfm_long_term_20d.pkl

格式：{'model': DummyModel(), 'scaler': DummyScaler()}
其中 DummyScaler.transform 为恒等映射，DummyModel.predict_classification
提供可序列化的替代实现。
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.dummy_models import DummyModel, DummyScaler  # noqa: E402

# 周期名 -> 天数
HORIZONS = {
    "short_term": 5,
    "mid_term": 10,
    "long_term": 20,
}


def create_placeholders(models_dir: Path) -> list:
    """在指定目录生成占位模型，返回落盘路径列表。

    抽成函数是为了让测试可生成到临时目录 —— 该脚本产物被 .gitignore 忽略，
    因此**不能假设** `models/` 下一定存在这些文件。
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for name, days in HORIZONS.items():
        path = models_dir / f"timesfm_{name}_{days}d.pkl"
        joblib.dump({"model": DummyModel(), "scaler": DummyScaler()}, path)
        written.append(path)

    # 为 ensemble 场景同时准备 lightgbm 占位（可选）
    for name, days in HORIZONS.items():
        path = models_dir / f"lightgbm_{name}_{days}d.pkl"
        joblib.dump({"model": DummyModel(), "scaler": DummyScaler()}, path)
        written.append(path)
    return written


def main() -> None:
    for path in create_placeholders(PROJECT_ROOT / "models"):
        print(f"written {path}")
