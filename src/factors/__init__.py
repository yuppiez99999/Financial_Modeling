"""多因子模型子包（Q2 路线：多因子模型集成，支持因子加权组合预测）。

对外契约：
  - FactorLibrary  : 因子库，把行情 DataFrame 扩展为因子矩阵
  - FactorModel    : 因子加权组合预测器（与 LightGBM / LSTM 同为可训练模型）
"""

from src.factors.factor_library import FactorLibrary, FACTOR_FAMILIES
from src.factors.factor_model import FactorModel
from src.factors.factor_predictor import FactorPredictor

__all__ = [
    "FactorLibrary",
    "FactorModel",
    "FactorPredictor",
    "FACTOR_FAMILIES",
]
