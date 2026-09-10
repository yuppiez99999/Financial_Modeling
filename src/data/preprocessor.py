"""特征工程与数据预处理（含技术指标）.

为推理引擎提供：
- transform(df, horizon_days): 在行情数据上计算技术指标特征。
- get_feature_columns(df_features, horizon_days): 返回特征列集合。
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """特征工程器。"""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config or {}
        feat_cfg = config.get("features", {}) if config else {}
        self.ma_windows = feat_cfg.get("technical", {}).get("ma_windows", [5, 10, 20, 60])
        self.rsi_window = feat_cfg.get("technical", {}).get("rsi_window", 14)

    # ------------------------------------------------------------------
    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标。"""
        out = df.copy()
        close = out["close"]

        # 均线
        for w in self.ma_windows:
            out[f"ma_{w}"] = close.rolling(w).mean()
            out[f"ma_{w}_ratio"] = close / out[f"ma_{w}"] - 1

        # 收益率
        out["ret_1"] = close.pct_change(1)
        out["ret_5"] = close.pct_change(5)
        out["ret_10"] = close.pct_change(10)

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(self.rsi_window).mean()
        loss = (-delta.clip(upper=0)).rolling(self.rsi_window).mean()
        rs = gain / (loss + 1e-9)
        out["rsi"] = 100 - (100 / (1 + rs))

        # 波动率
        out["volatility"] = out["ret_1"].rolling(20).std()

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        out["macd"] = ema12 - ema26
        out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
        out["macd_hist"] = out["macd"] - out["macd_signal"]

        # 布林带
        mid = close.rolling(20).mean()
        std = close.rolling(20).std()
        out["boll_upper"] = mid + 2 * std
        out["boll_lower"] = mid - 2 * std
        out["boll_pos"] = (close - mid) / (2 * std + 1e-9)

        # 量价特征
        if "volume" in out.columns:
            out["volume_ratio"] = out["volume"] / out["volume"].rolling(20).mean()

        return out

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame, horizon_days: int = 5) -> pd.DataFrame:
        """输入原始行情 DataFrame，输出带特征的 DataFrame。"""
        out = self._compute_indicators(df)
        # 填充 NaN，避免推理行丢失
        out = out.ffill().fillna(0)
        return out

    # ------------------------------------------------------------------
    def get_feature_columns(self, df_features: pd.DataFrame, horizon_days: int = 5) -> list[str]:
        """返回特征列集合。

        排除原始行情列与 target_* 目标列：目标列一旦混入特征集即构成
        目标泄漏（训练/评估指标虚高），因此无论在哪个阶段调用都强制排除。
        """
        exclude = {"date", "open", "high", "low", "close", "volume"}
        cols = [
            c
            for c in df_features.columns
            if c not in exclude and not str(c).startswith("target_")
        ]
        return cols

    def create_target(self, df: pd.DataFrame, horizon_days: int = 5) -> pd.DataFrame:
        """构造二分类目标列 target_{h}d：未来 h 日收盘收益 > 0 → 1。

        必须逐标的调用：多标的 concat 后统一 shift(-h) 会跨标的取到其他
        标的价格，使目标退化为"标的身份"信号（见 trainer._split_data 注释）。
        尾部 horizon_days 行无未来数据，目标为 NaN，由调用方统一 dropna。
        """
        out = df.copy()
        fwd_ret = out["close"].shift(-int(horizon_days)) / out["close"] - 1
        target = (fwd_ret > 0).astype(float)
        target[fwd_ret.isna()] = np.nan
        out[f"target_{int(horizon_days)}d"] = target
        return out


class DataPreprocessor:
    """训练用预处理器：持有 FeatureEngineer，提供 process(all_data) 契约。

    ModelTrainer 依赖（trainer.py）：
      - self.feature_engineer.create_target(df, horizon)  # 逐标的构造目标
      - self.feature_engineer.get_feature_columns(df, horizon)
      - self.process({symbol: raw_df}) → {symbol: feature_df}
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config or {}
        self.feature_engineer = FeatureEngineer(self.config)

    # ------------------------------------------------------------------
    def process(self, all_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """对每只标的做特征工程。

        不在此处构造目标列：目标由 trainer._split_data 按预测周期逐标的
        shift 构造（避免跨标的污染），本方法仅负责特征变换。
        """
        processed: dict[str, pd.DataFrame] = {}
        for symbol, df in all_data.items():
            try:
                processed[symbol] = self.feature_engineer.transform(df.copy())
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[preprocess] {symbol} 特征工程失败，跳过: {e}")
        logger.info(f"[preprocess] 完成 {len(processed)}/{len(all_data)} 标的特征工程")
        return processed
