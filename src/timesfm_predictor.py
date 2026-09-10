"""TimesFM 金融时序预测器包装（可选依赖）。

真实启用需要安装 `timesfm[torch]` 与系统兼容的 PyTorch。
为了兼容测试与未安装 PyTorch 的环境，本模块提供：
- 懒加载（Lazy import）timesfm / torch；
- 在导入/初始化失败时自动回退为占位 DummyModel，
  保证 PredictionEngine.predict() 在无 PyTorch 环境下正常工作并通过测试。
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class TimesFMFinancePredictor:
    """TimesFM 金融预测器。

    将 TimesFM 零样本时序模型包装为可供 PredictionEngine 直接调用的
    二分类预测接口（prediction / probability）。
    """

    def __init__(
        self,
        horizon_days: int = 5,
        context_days: int = 252,
        verbose: bool = False,
    ) -> None:
        self.horizon_days = int(horizon_days)
        self.context_days = int(context_days)
        self.verbose = verbose
        self._model = None
        self._backend = "placeholder"

    # ------------------------------------------------------------------
    # 内部：懒加载真实 TimesFM 模型
    # ------------------------------------------------------------------
    def _ensure_model(self) -> bool:
        """尝试加载真实 TimesFM；失败时回退占位并返回 False。"""
        if self._model is not None:
            return self._backend == "real"

        try:
            # 懒加载可选依赖，避免在未安装 torch 的测试环境强制导入
            from timesfm import TimesFM  # type: ignore

            from src.utils.dummy_models import DummyModel  # noqa: F401

            self._model = TimesFM()
            self._backend = "real"
            logger.info("TimesFM 已实例化（真实后端）")
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "TimesFM 真实后端不可用（%s），回退为占位模型 DummyModel", e
            )
            from src.utils.dummy_models import DummyModel

            self._model = DummyModel()
            self._backend = "placeholder"
            return False

    # ------------------------------------------------------------------
    # 辅助：提取收盘价序列（兼容 list / np.ndarray / pd.Series / DataFrame）
    # ------------------------------------------------------------------
    @staticmethod
    def _to_series(values: Any) -> np.ndarray:
        if values is None:
            return np.array([])
        if hasattr(values, "values"):
            values = values.values
        arr = np.asarray(values, dtype=float)
        return arr.reshape(-1) if arr.ndim > 1 else arr

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------
    def predict_classification(
        self,
        close_prices: Any,
        horizon: int | None = None,
        symbol: str = "",
    ) -> dict[str, Any]:
        """对收盘价序列进行方向预测（看涨/看跌）。

        Args:
            close_prices: 历史收盘价序列（list / ndarray / pd.Series）。
            horizon: 预测周期（缺省用构造时指定的 horizon_days）。
            symbol: 标的代码，用于结果回填。

        Returns:
            dict: 包含 prediction / probability / direction 等字段。
        """
        horizon = int(horizon) if horizon is not None else self.horizon_days
        series = self._to_series(close_prices)

        if series.size < 2:
            return {
                "prediction": 0,
                "probability": 0.5,
                "direction": "看跌",
                "symbol": symbol,
                "horizon": horizon,
                "error": "序列过短，无法预测",
            }

        self._ensure_model()

        if self._backend == "real":
            # 真实后端需要 TimesFM 的输入格式；此处适配其典型调用。
            # 若真实模型调用异常，则回退占位逻辑。
            try:
                import torch  # type: ignore

                context = series[-self.context_days :] if series.size > self.context_days else series
                input_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    forecast = self._model.forecast(input_tensor, horizon=horizon)
                fc = float(forecast[-1, -1].item()) if hasattr(forecast, "shape") else float(forecast)
                last_close = float(series[-1])
                # 以预测均值是否高于末值判定方向
                pred = 1 if fc > last_close else 0
                return {
                    "prediction": pred,
                    "probability": round(float(fc) / (last_close + 1e-9) / 2 + 0.5, 4),
                    "direction": "看涨" if pred == 1 else "看跌",
                    "symbol": symbol,
                    "horizon": horizon,
                    "backend": self._backend,
                }
            except Exception as e:  # noqa: BLE001
                logger.warning("TimesFM 真实推理失败（%s），回退占位", e)

        # 占位后端
        res = self._model.predict_classification(series, horizon=horizon, symbol=symbol)
        res["backend"] = self._backend
        return res

    def forecast(self, close_prices: Any, horizon: int | None = None) -> np.ndarray:
        """返回后市预测值序列（占位实现返回等长 0 序列）。"""
        horizon = int(horizon) if horizon is not None else self.horizon_days
        self._ensure_model()
        return np.zeros(horizon, dtype=float)
