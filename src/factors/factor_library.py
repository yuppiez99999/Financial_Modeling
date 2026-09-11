"""因子库：把原始行情扩展为**带因果方向的因子矩阵**。

与 `src.data.indicators` 的区别（两者互补，不重复造轮子）：

- `indicators` 产出的是**指标**：只要数值，不关心方向与预测含义；
- 本模块产出的是**因子**：每个因子都有明确的经济含义与**正向预期**
  （factor 值越大 → 未来上涨概率越高）。这是"因子加权组合预测"的前提，
  否则对原始指标直接加权在方向上毫无意义（例如 ADX 越大并不代表越看涨）。

因子族（family）：
  - trend       趋势族：均线偏离、EMA 斜率、ADX 趋势强度
  - momentum    动量族：RSI 等权打分、ROC、MACD 柱
  - volatility  波动族：历史波动率、ATR 相对值（低波动溢价）
  - volume      量能族：量比、OBV 斜率、CMF
  - reversal    反转族：布林 %B 反向、KDJ 超买超卖打分
  - sentiment   情绪族：新闻情感归一化得分（可选）
  - macro       宏观族：宏观指标 z-score 等权合成（可选）

设计约束：
- **只用历史数据**：全部因子基于 rolling / ewm / shift，天然无前视；
- **离线可跑**：不触网（情绪/宏观因子数据缺失时返回中性 0，不编造）；
- 输出列统一为 `factor_*` 前缀，便于与训练特征区分。
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 因子族 -> 该族下的因子列（family 权重在 FactorModel 中归一化）
FACTOR_FAMILIES: dict[str, tuple[str, ...]] = {
    "trend": ("factor_ma_bias", "factor_ema_slope", "factor_adx_strength"),
    "momentum": ("factor_rsi_score", "factor_roc", "factor_macd_hist"),
    "volatility": ("factor_volatility", "factor_atr_ratio"),
    "volume": ("factor_volume_ratio", "factor_obv_slope", "factor_cmf"),
    "reversal": ("factor_boll_revert", "factor_kdj_score"),
    "sentiment": ("factor_sentiment",),
    "macro": ("factor_macro",),
}


def _zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """滚动 z-score（因果，仅用窗口内历史），窗口内无波动时归零。"""
    mean = series.rolling(window, min_periods=5).mean()
    std = series.rolling(window, min_periods=5).std()
    return ((series - mean) / std.replace(0, np.nan)).fillna(0.0)


def _tanh_clip(series: pd.Series) -> pd.Series:
    """把任意量纲压到 (-1, 1)，保证不同因子可直接加权（有界、抗极值）。"""
    return np.tanh(series.replace([np.inf, -np.inf], np.nan).fillna(0.0))


class FactorLibrary:
    """因子库：`compute(df)` → 附加 factor_* 列的 DataFrame。"""

    NAME = "FactorLibrary"
    VERSION = "1.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = ((config or {}).get("model", {}) or {}).get("factors", {}) or {}
        self.enabled_families: list[str] = list(cfg.get("families", list(FACTOR_FAMILIES)))
        self.zscore_window = int(cfg.get("zscore_window", 252))
        smoothing = int(cfg.get("smoothing", 5))

        ind_cfg = ((config or {}).get("features", {}) or {}).get("technical", {}) or {}
        self.rsi_window = int(cfg.get("rsi_window", ind_cfg.get("rsi_window", 14)))
        self.smoothing = max(smoothing, 1)

    # ------------------------------------------------------------------
    # 各族因子
    # ------------------------------------------------------------------
    def _trend(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        out = df.copy()
        # 均线偏离：价格高于均线 → 趋势向上（正）
        ma = close.rolling(20, min_periods=5).mean()
        out["factor_ma_bias"] = _tanh_clip((close / ma - 1.0) * 10.0)
        # EMA 斜率：短期均线相对长期均线的变化率（正 → 上行）
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        out["factor_ema_slope"] = _tanh_clip((ema_fast / ema_slow - 1.0) * 15.0)
        # ADX 趋势强度：需要方向配合，用 +DI/-DI 之差定向（正 → 多头趋势）
        up_move = df["high"].diff()
        down_move = -df["low"].diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - close.shift()).abs(),
            (df["low"] - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.rsi_window, min_periods=5).mean().replace(0, np.nan)
        plus_di = 100 * plus_dm.rolling(self.rsi_window, min_periods=5).mean() / atr
        minus_di = 100 * minus_dm.rolling(self.rsi_window, min_periods=5).mean() / atr
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = (100 * dx).rolling(self.rsi_window, min_periods=5).mean()
        direction = np.sign((plus_di - minus_di).fillna(0.0))
        out["factor_adx_strength"] = _tanh_clip(direction * (adx.fillna(0.0) / 50.0) * 2.0)
        return out

    def _momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        out = df.copy()
        # RSI 等权打分：50 为中性，(RSI-50)/50 ∈ [-1,1]
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(self.rsi_window, min_periods=5).mean()
        loss = (-delta.clip(upper=0)).rolling(self.rsi_window, min_periods=5).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        out["factor_rsi_score"] = _tanh_clip((rsi - 50.0) / 25.0)
        # ROC：20 日动量
        out["factor_roc"] = _tanh_clip(close.pct_change(20) * 10.0)
        # MACD 柱（正 → 多头动能增强）
        macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        macd_hist = macd - macd.ewm(span=9, adjust=False).mean()
        out["factor_macd_hist"] = _tanh_clip(macd_hist / (close.abs() + 1e-9) * 200.0)
        return out

    def _volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        out = df.copy()
        ret = close.pct_change()
        vol = ret.rolling(20, min_periods=5).std()
        # 低波动溢价：波动越高 → 因子越负（负向预期）
        out["factor_volatility"] = _tanh_clip(-_zscore(vol, self.zscore_window))
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - close.shift()).abs(),
            (df["low"] - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.rsi_window, min_periods=5).mean()
        out["factor_atr_ratio"] = _tanh_clip(_zscore(-(atr / (close.abs() + 1e-9)), self.zscore_window))
        return out

    def _volume(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "volume" not in df.columns:
            for col in FACTOR_FAMILIES["volume"]:
                out[col] = 0.0
            return out
        volume = df["volume"].astype(float)
        # 量比：放量配合上行视为正
        vol_ma = volume.rolling(20, min_periods=5).mean().replace(0, np.nan)
        out["factor_volume_ratio"] = _tanh_clip((volume / vol_ma - 1.0).fillna(0.0))
        # OBV 斜率（归一化）
        obv = (np.sign(df["close"].diff().fillna(0.0)) * volume).cumsum()
        out["factor_obv_slope"] = _tanh_clip(_zscore(obv.diff(5).fillna(0.0), self.zscore_window))
        # CMF：蔡金资金流（正 → 资金流入）
        mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
            (df["high"] - df["low"]).replace(0, np.nan)
        )
        mfv = (mfm.fillna(0.0) * volume).rolling(20, min_periods=5).sum()
        out["factor_cmf"] = _tanh_clip(mfv / volume.rolling(20, min_periods=5).sum().replace(0, np.nan))
        return out

    def _reversal(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        out = df.copy()
        # 布林 %B 反向：超买（高位）→ 均值回归看跌，故取负
        mid = close.rolling(20, min_periods=5).mean()
        std = close.rolling(20, min_periods=5).std()
        pct_b = (close - mid) / (2 * std + 1e-9)
        out["factor_boll_revert"] = _tanh_clip(-pct_b)
        # KDJ 超买超卖打分（反向）
        low_n = df["low"].rolling(9, min_periods=3).min()
        high_n = df["high"].rolling(9, min_periods=3).max()
        rsv = (close - low_n) / (high_n - low_n).replace(0, np.nan) * 100
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3 * k - 2 * d
        out["factor_kdj_score"] = _tanh_clip(-(j.fillna(50.0) - 50.0) / 50.0)
        return out

    def _sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """情绪因子：复用已有情感特征（列名见 sentiment_analyzer）。

        数据缺失时不编造，直接给中性 0（与宏观零值口径一致）。
        """
        out = df.copy()
        candidates = ["sentiment_score", "sentiment", "news_sentiment", "sentiment_mean"]
        value = None
        for col in candidates:
            if col in df.columns:
                value = pd.to_numeric(df[col], errors="coerce")
                break
        if value is None:
            out["factor_sentiment"] = 0.0
        else:
            out["factor_sentiment"] = _tanh_clip(value.fillna(0.0))
        return out

    def _macro(self, df: pd.DataFrame) -> pd.DataFrame:
        """宏观因子：把已有宏观特征列做 z-score 后等权合成（缺失 → 0）。"""
        out = df.copy()
        macro_cols = [c for c in df.columns if str(c).startswith("macro_")]
        if not macro_cols:
            out["factor_macro"] = 0.0
            return out
        scores = []
        for col in macro_cols:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.abs().sum() == 0:
                continue  # 显式零值（无数据）不参与合成
            scores.append(_zscore(series, self.zscore_window))
        out["factor_macro"] = (
            _tanh_clip(pd.concat(scores, axis=1).mean(axis=1)) if scores else 0.0
        )
        if not scores:
            out["factor_macro"] = 0.0
        return out

    # ------------------------------------------------------------------
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算全部启用因子族的因子列。"""
        required = {"close", "high", "low"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"因子计算缺少行情列: {sorted(missing)}")

        handlers = {
            "trend": self._trend,
            "momentum": self._momentum,
            "volatility": self._volatility,
            "volume": self._volume,
            "reversal": self._reversal,
            "sentiment": self._sentiment,
            "macro": self._macro,
        }
        out = df.copy()
        for family in self.enabled_families:
            handler = handlers.get(family)
            if handler is None:
                logger.warning("[factors] 未知因子族 %s，跳过", family)
                continue
            try:
                out = handler(out)
            except Exception as e:  # noqa: BLE001
                # fail-soft：单个因子族失败只置中性 0，不阻断整条链路
                logger.warning("[factors] 因子族 %s 计算失败，置中性: %s", family, e)
                for col in FACTOR_FAMILIES.get(family, ()):
                    out[col] = 0.0

        # 平滑 + 清理：消除单日噪音，保证无 NaN / inf
        for col in self.factor_columns(out):
            series = pd.to_numeric(out[col], errors="coerce")
            if self.smoothing > 1:
                series = series.rolling(self.smoothing, min_periods=1).mean()
            out[col] = series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
        return out

    @staticmethod
    def factor_columns(df: pd.DataFrame) -> list[str]:
        """返回因子列（family 顺序稳定，便于复现）。"""
        ordered: list[str] = []
        for cols in FACTOR_FAMILIES.values():
            for col in cols:
                if col in df.columns and col not in ordered:
                    ordered.append(col)
        for col in df.columns:
            if str(col).startswith("factor_") and col not in ordered:
                ordered.append(col)
        return ordered

    def family_map(self, factor_cols: list[str] | None = None) -> dict[str, list[str]]:
        """返回 {family: [factor_cols]}（仅保留实际存在的列）。"""
        if factor_cols is None:
            return {k: list(v) for k, v in FACTOR_FAMILIES.items()}
        cols = set(factor_cols)
        return {k: [c for c in v if c in cols] for k, v in FACTOR_FAMILIES.items()}
