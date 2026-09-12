"""兼容性再导出（Q4 核心实现已迁移至 :mod:`src.trading.risk_advisor`）。

历史版本使用 ``risk_advice`` 命名；``risk_advisor`` 为其重构版，统一复用
``src.data.indicators.compute_atr``。本文件仅作再导出，保持既有 import 路径
（main.py / server.py / health_report.py / 测试）不变，单一事实源在 ``risk_advisor``。
"""
from src.trading.risk_advisor import (  # noqa: F401
    STATUS_OK,
    STATUS_WITHHELD,
    STATUS_UNAVAILABLE,
    DISCLAIMER,
    RiskAdvisor,
    StopTakePlan,
)

__all__ = [
    "STATUS_OK",
    "STATUS_WITHHELD",
    "STATUS_UNAVAILABLE",
    "DISCLAIMER",
    "RiskAdvisor",
    "StopTakePlan",
]
