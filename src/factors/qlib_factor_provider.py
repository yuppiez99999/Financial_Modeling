"""qlib Alpha158 因子供给器（S13 / G3，T13.2）：把 Alpha158 喂进现有因子体系。

设计定位（Issue #29 集成方案 G3）：
  qlib 的 Alpha158 是 98 列「原始横截面因子」，本项目 FactorLibrary 输出的是
  **带因果方向、有界 (-1,1) 的组合因子**（15 个）。两者不是替代关系：
  Alpha158 需要一个供给器把它**降维成与 FactorLibrary 同构的形态**，
  才能进现有 factor_library / factor_model 加权链路而不破坏口径。

本模块做什么：
  - ``QlibFactorProvider.compute(df)``：对单标的行情计算 Alpha158，再按族
    （动量/趋势/波动/量价/位置）压缩成少量 ``factor_a158_*`` 有界因子列；
  - 压缩方式 = 逐族 |IC| 不可用时退化为**等权符号合成**（无前视：只用
    当日因子值，权重不依赖未来收益）；
  - 输出列与 FactorLibrary 的 ``factor_*`` 命名空间隔离（``factor_a158_`` 前缀），
    避免与现有 15 因子撞名。

本模块**不做什么**（边界）：
  - **不改门禁**：是否纳入主线由 T13.3 增量验证 + T13.4 人工检查点决定；
  - **不引入 qlib 运行时依赖**：内部只调 ``integrations.qlib.alpha158``
    （纯 pandas 实现），qlib 安装与否不影响 CI；
  - **不编造**：行情不足（< MIN_ROWS）时返回中性 0 因子，并显式标注
    ``a158_available=False`` 列（下游可过滤，而非误当真值）。
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 与 integrations/qlib/alpha158.py 一致的最少历史
MIN_ROWS = 60

# 族定义：Alpha158 列名（含窗口）→ 经济含义分族。
# 方向（正/负向预期）与 FactorLibrary 各族的因果方向声明对齐。
A158_FAMILIES: dict[str, dict[str, Any]] = {
    # 动量族：ROC 为过去价格 / 现价 - 1 → 取负号才与「动量向上=正」一致
    "momentum": {
        "prefixes": ("A158_ROC",),
        "sign": -1.0,
    },
    # 趋势族：MA/RSV/RANK 高 → 价格处在上行区间（正向）
    "trend": {
        "prefixes": ("A158_MA", "A158_RSV", "A158_RANK"),
        "sign": 1.0,
    },
    # 波动族：STD/WVMA 高 → 波动大（低波动溢价，负向）
    "volatility": {
        "prefixes": ("A158_STD", "A158_WVMA", "A158_VSTD"),
        "sign": -1.0,
    },
    # 量价族：CORR(价,量) 为正 → 量价配合（正向）；CNTP 高 → 上涨日占比高
    "volume": {
        "prefixes": ("A158_CORR", "A158_CNTP", "A158_VMA", "A158_SUMP", "A158_VSUMP"),
        "sign": 1.0,
    },
    # 位置族：QTLU 高 → 价格靠近区间上沿（均值回归，负向）；QTLD 反向
    "reversal": {
        "prefixes": ("A158_QTLU", "A158_KUP", "A158_KUP2"),
        "sign": -1.0,
        "inverse_prefixes": ("A158_QTLD", "A158_KLOW", "A158_KLOW2"),
    },
}

PROVIDER_FACTOR_COLUMNS = tuple(
    f"factor_a158_{name}" for name in A158_FAMILIES
)  # 注意：不含 factor_a158_available 哨兵列


def _tanh_clip(series: pd.Series) -> pd.Series:
    """与 FactorLibrary._tanh_clip 同款：压到 (-1,1)，抗极值。"""
    return np.tanh(series.replace([np.inf, -np.inf], np.nan).fillna(0.0))


def _family_score(factors: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    """把一族 Alpha158 列压缩成单一有界得分（等权，无未来信息）。

    各列先做**滚动 z-score**（只用过去 zscore_window 行）再取均值 ——
    不做 z-score 直接平均会让量纲大的列（如 MA≈股价量级）淹没量纲小的列。
    """
    cols: list[str] = []
    for prefix in spec["prefixes"]:
        cols.extend(c for c in factors.columns if c.startswith(prefix))
    for prefix in spec.get("inverse_prefixes", ()):
        cols.extend(c for c in factors.columns if c.startswith(prefix))
    if not cols:
        return pd.Series(0.0, index=factors.index)

    scores: list[pd.Series] = []
    for col in cols:
        s = pd.to_numeric(factors[col], errors="coerce")
        # 滚动 z-score（因果）：窗口内标准化，样本不足置 0
        mu = s.rolling(120, min_periods=20).mean()
        sd = s.rolling(120, min_periods=20).std()
        z = ((s - mu) / sd.replace(0, np.nan)).fillna(0.0)
        sign = spec["sign"]
        if "inverse_prefixes" in spec and any(
            col.startswith(p) for p in spec["inverse_prefixes"]
        ):
            sign = -spec["sign"]  # inverse 前缀反向（QTLD 低 → 看涨）
        scores.append(z * sign)
    return pd.concat(scores, axis=1).mean(axis=1)


class QlibFactorProvider:
    """Alpha158 → factor_a158_* 因子供给器（与 FactorLibrary.compute 同构）。"""

    NAME = "QlibFactorProvider"
    VERSION = "1.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        qlib_cfg = ((self.config.get("model", {}) or {}).get("factors", {}) or {}).get(
            "qlib_alpha158", {}) or {}
        self.enabled: bool = bool(qlib_cfg.get("enabled", False))
        self.families: list[str] = list(qlib_cfg.get("families", list(A158_FAMILIES)))

    # ------------------------------------------------------------------
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """对单标的行情追加 ``factor_a158_*`` 列（缺数据 → 中性 0，不猜）。"""
        out = df.copy()
        required = {"open", "high", "low", "close", "volume"}
        usable = required.issubset(out.columns) and len(out) >= MIN_ROWS

        if not usable:
            # 如实标注：a158_available=False 供下游过滤（不是有效因子）
            out["factor_a158_available"] = 0.0
            for col in PROVIDER_FACTOR_COLUMNS:
                out[col] = 0.0
            return out

        try:
            from integrations.qlib.alpha158 import compute_alpha158

            factors = compute_alpha158(out)
        except Exception as e:  # noqa: BLE001 - fail-soft：供给器失败不阻断主链路
            logger.warning("[qlib-factors] Alpha158 计算失败，置中性: %s", e)
            out["factor_a158_available"] = 0.0
            for col in PROVIDER_FACTOR_COLUMNS:
                out[col] = 0.0
            return out

        out["factor_a158_available"] = 1.0
        for name, spec in A158_FAMILIES.items():
            col = f"factor_a158_{name}"
            if name not in self.families:
                out[col] = 0.0
                continue
            try:
                out[col] = _tanh_clip(_family_score(factors, spec))
            except Exception as e:  # noqa: BLE001 - 单族失败置中性
                logger.warning("[qlib-factors] 族 %s 压缩失败，置中性: %s", name, e)
                out[col] = 0.0
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def factor_columns(df: pd.DataFrame) -> list[str]:
        """返回 DataFrame 中的 factor_a158_* 列（顺序稳定）。"""
        return [c for c in PROVIDER_FACTOR_COLUMNS if c in df.columns]

    def family_map(self, factor_cols: list[str] | None = None) -> dict[str, list[str]]:
        """返回 {family: [factor_cols]}（与 FactorLibrary.family_map 同构）。"""
        cols = set(factor_cols) if factor_cols is not None else set(PROVIDER_FACTOR_COLUMNS)
        return {
            name: [f"factor_a158_{name}"] if f"factor_a158_{name}" in cols else []
            for name in A158_FAMILIES
        }
