"""占位模型与缩放器，用于在无 PyTorch / TimesFM 场景下测试与运行。

这些类保持"可序列化"（pickle 友好），确保在没有安装 torch 的环境中
也能通过 joblib 保存/加载占位 pickle 文件，并让 PredictionEngine 正常工作。
"""
from __future__ import annotations

from typing import Any

import numpy as np


class DummyScaler:
    """恒等映射缩放器。

    与 sklearn scaler 对齐提供 transform / fit_transform / fit /
    inverse_transform 接口，其中 transform 为恒等映射。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def fit(self, X: Any, y: Any = None) -> "DummyScaler":
        return self

    def transform(self, X: Any) -> Any:
        return X

    def fit_transform(self, X: Any, y: Any = None) -> Any:
        return X

    def inverse_transform(self, X: Any) -> Any:
        return X


class DummyModel:
    """可序列化的占位分类模型。

    提供与真实模型（如 LightGBM / TimesFM）兼容的接口：
    - predict(X)  -> 返回 0/1 标签数组
    - predict_proba(X) -> 返回 [[p0, p1]] 概率数组（默认 0.5/0.5）
    - predict_classification(series, horizon, symbol) -> 返回 dict
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.classes_ = np.array([0, 1])
        # 对齐 sklearn 以支持 predictor 中按 n_features_in_ 截断/填充
        self.n_features_in_: int | None = None

    def predict(self, X: Any) -> np.ndarray:
        if hasattr(X, "shape"):
            n = X.shape[0]
        else:
            n = len(X)
        return np.zeros(n, dtype=int)

    def predict_proba(self, X: Any) -> np.ndarray:
        if hasattr(X, "shape"):
            n = X.shape[0]
        else:
            n = len(X)
        # 默认给出中性概率 0.5/0.5
        return np.full((n, 2), 0.5)

    def predict_classification(
        self,
        series: Any,
        horizon: int = 5,
        symbol: str = "",
    ) -> dict[str, Any]:
        """TimesFM 风格的占位预测接口。"""
        return {
            "prediction": 0,
            "probability": 0.5,
            "direction": "看跌",
            "symbol": symbol,
            "horizon": horizon,
            "note": "dummy placeholder prediction",
        }
