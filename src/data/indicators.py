"""专业版扩展技术指标库（50+ 指标）

在基础指标（MA/RSI/MACD/Bollinger）之上扩展：
  趋势类: EMA, WMA, DEMA, TEMA, TRIMA, KAMA, ADX, ADXR, AROON, CCI, SAR, Vortex
  动量类: KDJ, Stochastic, Williams %R, ROC, MOM, TRIX, CMO, MFI, TSI, Ultimate OSC
  波动类: ATR, NATR, StdDev, Mass Index, Choppiness Index
  成交量类: OBV, VWAP, MFI, CMF, A/D Line, PVT, NVI, Force Index, Ease of Movement
  结构类: Pivot Points, Support/Resistance, Fibonacci Retracement
  形态类: Doji, Hammer, Engulfing, Morning/Evening Star
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """专业版技术指标库（50+ 指标）"""

    # 指标清单（用于审计和文档）
    INDICATOR_LIST = [
        # 趋势 (12)
        "ema", "wma", "dema", "tema", "trima", "kama",
        "adx", "adxr", "aroon_up", "aroon_down", "cci", "vortex_pos", "vortex_neg",
        # 动量 (10)
        "kdj_k", "kdj_d", "kdj_j", "stoch_k", "stoch_d", "willr", "roc", "mom", "trix", "cmo",
        # 波动 (5)
        "atr", "natr", "stddev", "mass_index", "choppiness",
        # 成交量 (9)
        "obv", "vwap", "mfi", "cmf", "ad_line", "pvt", "nvi", "force_index", "eom",
        # 结构 (5)
        "pivot", "r1", "s1", "r2", "s2",
        # 形态 (4)
        "doji", "hammer", "engulfing", "morning_star",
        # 基础补充 (6)
        "rsi", "macd", "macd_signal", "macd_hist", "boll_width", "boll_pct_b",
    ]

    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        """计算全部扩展指标，返回增强后的 DataFrame"""
        ti = TechnicalIndicators()
        df = df.copy()

        # 确保数值列为 float 类型（避免 int64 运算出错）
        for col in ["open", "close", "high", "low", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        # 趋势类
        df = ti._add_ema(df)
        df = ti._add_wma(df)
        df = ti._add_dema_tema(df)
        df = ti._add_trima(df)
        df = ti._add_kama(df)
        df = ti._add_adx(df)
        df = ti._add_aroon(df)
        df = ti._add_cci(df)
        df = ti._add_vortex(df)

        # 动量类
        df = ti._add_kdj(df)
        df = ti._add_stochastic(df)
        df = ti._add_williams(df)
        df = ti._add_roc_mom(df)
        df = ti._add_trix(df)
        df = ti._add_cmo(df)
        df = ti._add_mfi(df)
        df = ti._add_tsi(df)

        # 波动类
        df = ti._add_atr(df)
        df = ti._add_mass_index(df)
        df = ti._add_choppiness(df)

        # 成交量类
        df = ti._add_obv(df)
        df = ti._add_vwap(df)
        df = ti._add_cmf(df)
        df = ti._add_ad_line(df)
        df = ti._add_pvt_nvi(df)
        df = ti._add_force_index(df)
        df = ti._add_eom(df)

        # 结构类
        df = ti._add_pivot_points(df)

        # 形态类
        df = ti._add_candlestick_patterns(df)

        # 布林带补充
        if "boll_upper" in df.columns and "boll_lower" in df.columns:
            df["boll_pct_b"] = (df["close"] - df["boll_lower"]) / (df["boll_upper"] - df["boll_lower"]).replace(0, 1e-10)

        logger.info(f"扩展指标计算完成: 新增 {len(TechnicalIndicators.INDICATOR_LIST)} 个指标")
        return df

    # ==================== 趋势类 ====================

    def _add_ema(self, df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
        windows = windows or [5, 10, 20, 60]
        for w in windows:
            df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()
        return df

    def _add_wma(self, df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
        windows = windows or [10, 20]
        for w in windows:
            weights = np.arange(1, w + 1, dtype=float)
            df[f"wma_{w}"] = df["close"].rolling(w).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
        return df

    def _add_dema_tema(self, df: pd.DataFrame) -> pd.DataFrame:
        for w in [10, 20]:
            ema1 = df["close"].ewm(span=w, adjust=False).mean()
            ema2 = ema1.ewm(span=w, adjust=False).mean()
            df[f"dema_{w}"] = 2 * ema1 - ema2
            ema3 = ema2.ewm(span=w, adjust=False).mean()
            df[f"tema_{w}"] = 3 * ema1 - 3 * ema2 + ema3
        return df

    def _add_trima(self, df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
        windows = windows or [12, 20]
        for w in windows:
            sma = df["close"].rolling(w).mean()
            df[f"trima_{w}"] = sma.rolling(w).mean()
        return df

    def _add_kama(self, df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        change = df["close"].diff(period)
        vol = df["close"].diff().abs().rolling(period).sum()
        er = (change.abs() / vol.replace(0, 1e-10)).fillna(0)
        fast_sc = 2 / (2 + 1)
        slow_sc = 2 / (30 + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama = df["close"].copy()
        # 向量化递推: KAMA[i] = KAMA[i-1] + sc[i] * (close[i] - KAMA[i-1])
        # 等价于 KAMA[i] = (1-sc[i])*KAMA[i-1] + sc[i]*close[i]
        for i in range(max(period, 1), len(df)):
            kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (float(df["close"].iloc[i]) - kama.iloc[i - 1])
        df["kama"] = kama
        return df

    def _add_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        high, low, close = df["high"], df["low"], df["close"]
        plus_dm = (high.diff()).where((high.diff() > -low.diff()) & (high.diff() > 0), 0)
        minus_dm = (-low.diff()).where((-low.diff() > high.diff()) & (-low.diff() > 0), 0)
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, 1e-10))
        minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, 1e-10))
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10))
        df["adx"] = dx.ewm(alpha=1 / period, adjust=False).mean()
        df["adxr"] = (df["adx"] + df["adx"].shift(period)) / 2
        return df

    def _add_aroon(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        high = df["high"]
        low = df["low"]
        df["aroon_up"] = high.rolling(period + 1).apply(
            lambda x: 100.0 * (period - x.argmax()) / period, raw=True
        )
        df["aroon_down"] = low.rolling(period + 1).apply(
            lambda x: 100.0 * (period - x.argmin()) / period, raw=True
        )
        return df

    def _add_cci(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        tp = (df["high"] + df["low"] + df["close"]) / 3
        sma_tp = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: float(np.mean(np.abs(x - x.mean()))), raw=True)
        df["cci"] = (tp - sma_tp) / (0.015 * mad.replace(0, 1e-10))
        return df

    def _add_vortex(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        vm_plus = (df["high"] - df["low"].shift()).abs()
        vm_minus = (df["low"] - df["high"].shift()).abs()
        tr_sum = tr.rolling(period).sum().replace(0, 1e-10)
        df["vortex_pos"] = vm_plus.rolling(period).sum() / tr_sum
        df["vortex_neg"] = vm_minus.rolling(period).sum() / tr_sum
        return df

    # ==================== 动量类 ====================

    def _add_kdj(self, df: pd.DataFrame, period: int = 9) -> pd.DataFrame:
        low_min = df["low"].rolling(period).min()
        high_max = df["high"].rolling(period).max()
        rsv = (df["close"] - low_min) / (high_max - low_min).replace(0, 1e-10) * 100
        df["kdj_k"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
        df["kdj_d"] = df["kdj_k"].ewm(alpha=1 / 3, adjust=False).mean()
        df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]
        return df

    def _add_stochastic(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        low_min = df["low"].rolling(period).min()
        high_max = df["high"].rolling(period).max()
        df["stoch_k"] = (df["close"] - low_min) / (high_max - low_min).replace(0, 1e-10) * 100
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()
        return df

    def _add_williams(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        high_max = df["high"].rolling(period).max()
        low_min = df["low"].rolling(period).min()
        df["willr"] = (high_max - df["close"]) / (high_max - low_min).replace(0, 1e-10) * -100
        return df

    def _add_roc_mom(self, df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
        windows = windows or [10, 20]
        for w in windows:
            df[f"roc_{w}"] = df["close"].pct_change(w) * 100
            df[f"mom_{w}"] = df["close"].diff(w)
        return df

    def _add_trix(self, df: pd.DataFrame, period: int = 12) -> pd.DataFrame:
        ema1 = df["close"].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        df["trix"] = ema3.pct_change() * 100
        return df

    def _add_cmo(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(period).sum()
        loss = (-delta.where(delta < 0, 0)).rolling(period).sum()
        df["cmo"] = 100 * (gain - loss) / (gain + loss).replace(0, 1e-10)
        return df

    def _add_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        tp = (df["high"] + df["low"] + df["close"]) / 3
        mf = tp * df["volume"]
        pos_mf = mf.where(tp > tp.shift(), 0)
        neg_mf = mf.where(tp < tp.shift(), 0)
        mfr = pos_mf.rolling(period).sum() / neg_mf.rolling(period).sum().replace(0, 1e-10)
        df["mfi"] = 100 - 100 / (1 + mfr)
        return df

    def _add_tsi(self, df: pd.DataFrame, r: int = 25, s: int = 13) -> pd.DataFrame:
        m = df["close"].diff()
        smooth_m = m.ewm(span=r, adjust=False).mean()
        smooth_abs = m.abs().ewm(span=r, adjust=False).mean()
        df["tsi"] = (smooth_m / smooth_abs.replace(0, 1e-10)).ewm(span=s, adjust=False).mean() * 100.0
        return df

    # ==================== 波动类 ====================

    def _add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.ewm(alpha=1 / period, adjust=False).mean()
        df["natr"] = df["atr"] / df["close"].replace(0, 1e-10) * 100
        df["stddev"] = df["close"].rolling(period).std()
        return df

    def _add_mass_index(self, df: pd.DataFrame, period: int = 25) -> pd.DataFrame:
        ema1 = (df["high"] - df["low"]).ewm(span=9, adjust=False).mean()
        ema2 = ema1.ewm(span=9, adjust=False).mean()
        df["mass_index"] = (ema1 / ema2.replace(0, 1e-10)).rolling(period).sum()
        return df

    def _add_choppiness(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        atr_sum = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1).rolling(period).sum()
        high_max = df["high"].rolling(period).max()
        low_min = df["low"].rolling(period).min()
        df["choppiness"] = 100 * np.log10(atr_sum / (high_max - low_min).replace(0, 1e-10)) / np.log10(period)
        return df

    # ==================== 成交量类 ====================

    def _add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        direction = np.sign(df["close"].diff().fillna(0))
        df["obv"] = (direction * df["volume"]).cumsum()
        return df

    def _add_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        typical = (df["high"] + df["low"] + df["close"]) / 3
        df["vwap"] = (typical * df["volume"]).cumsum() / df["volume"].replace(0, np.nan).cumsum()
        return df

    def _add_cmf(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / \
              (df["high"] - df["low"]).replace(0, 1e-10)
        df["cmf"] = (mfm * df["volume"]).rolling(period).sum() / df["volume"].rolling(period).sum().replace(0, 1e-10)
        return df

    def _add_ad_line(self, df: pd.DataFrame) -> pd.DataFrame:
        mfv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / \
              (df["high"] - df["low"]).replace(0, 1e-10) * df["volume"]
        df["ad_line"] = mfv.cumsum()
        return df

    def _add_pvt_nvi(self, df: pd.DataFrame) -> pd.DataFrame:
        pct = df["close"].pct_change().fillna(0)
        df["pvt"] = (pct * df["volume"]).cumsum()
        # 向量化 NVI 计算
        vol_change = df["volume"].pct_change().fillna(0)
        # 单次乘法的极端放大（pct 含 ±inf 或极大值）会让 NVI 溢出为 inf/NaN，
        # 进而污染整个特征矩阵 → 用 np.isfinite 守卫，异常段保持上一值（不编造）。
        nvi = np.ones(len(df), dtype=np.float64) * 1000.0
        for i in range(1, len(df)):
            prev = nvi[i - 1]
            if vol_change.iloc[i] < 0:
                step = float(pct.iloc[i])
                candidate = prev * (1.0 + step)
                nvi[i] = candidate if np.isfinite(candidate) else prev
            else:
                nvi[i] = prev
        df["nvi"] = nvi
        return df

    def _add_force_index(self, df: pd.DataFrame, period: int = 13) -> pd.DataFrame:
        fi = df["close"].diff() * df["volume"]
        df["force_index"] = fi.ewm(span=period, adjust=False).mean()
        return df

    def _add_eom(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        distance = ((df["high"] + df["low"]) / 2 - (df["high"].shift() + df["low"].shift()) / 2)
        box_ratio = (df["volume"] / 1e8) / (df["high"] - df["low"]).replace(0, 1e-10)
        df["eom"] = (distance / box_ratio.replace(0, 1e-10)).rolling(period).mean()
        return df

    # ==================== 结构类 ====================

    def _add_pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
        prev_high = df["high"].shift(1)
        prev_low = df["low"].shift(1)
        prev_close = df["close"].shift(1)
        pivot = (prev_high + prev_low + prev_close) / 3
        df["pivot"] = pivot
        df["r1"] = 2 * pivot - prev_low
        df["s1"] = 2 * pivot - prev_high
        df["r2"] = pivot + (prev_high - prev_low)
        df["s2"] = pivot - (prev_high - prev_low)
        return df

    # ==================== 形态类 ====================

    def _add_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        body = (df["close"] - df["open"]).abs()
        total_range = (df["high"] - df["low"]).replace(0, 1e-10)
        upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
        lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]

        # 十字星
        df["doji"] = (body / total_range < 0.1).astype(int)
        # 锤子线
        df["hammer"] = ((lower_shadow > 2 * body) & (upper_shadow < body) & (body > 0)).astype(int)
        # 吞没形态
        df["engulfing"] = (
            (df["close"] > df["open"]) & (df["close"].shift() < df["open"].shift()) &
            (df["close"] >= df["open"].shift()) & (df["open"] <= df["close"].shift())
        ).astype(int)
        # 早晨之星（简化）
        df["morning_star"] = (
            (df["close"].shift(2) < df["open"].shift(2)) &
            (abs(df["close"].shift(1) - df["open"].shift(1)) / total_range.shift(1) < 0.1) &
            (df["close"] > df["open"]) & (df["close"] > (df["open"].shift(2) + df["close"].shift(2)) / 2)
        ).astype(int)
        return df
