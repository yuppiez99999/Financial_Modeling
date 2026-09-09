"""金融市场预测模型 - 数据预处理与特征工程模块"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MacroIndicatorFetcher:
    """宏观指标获取器"""

    MACRO_INDICATORS = {
        "cpi": {"name": "CPI", "url": "https://api.macrotrends.net/assets/php/macroChartData.php", "params": {"m": 14}},
        "pmi": {"name": "PMI", "url": "https://api.macrotrends.net/assets/php/macroChartData.php", "params": {"m": 15}},
        "gdp": {"name": "GDP", "url": "https://api.macrotrends.net/assets/php/macroChartData.php", "params": {"m": 13}},
    }

    def __init__(self):
        self.cache: dict[str, pd.DataFrame] = {}

    def fetch_indicator(self, indicator: str) -> pd.DataFrame | None:
        """获取宏观指标数据"""
        if indicator in self.cache:
            return self.cache[indicator]

        try:
            import requests
            cfg = self.MACRO_INDICATORS.get(indicator)
            if not cfg:
                return None

            resp = requests.get(cfg["url"], params=cfg["params"], timeout=15)
            data = resp.json()
            dates = [d[0] for d in data["data"]]
            values = [d[1] for d in data["data"]]
            df = pd.DataFrame({"date": dates, f"macro_{indicator}": values})
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            df[f"macro_{indicator}"] = pd.to_numeric(df[f"macro_{indicator}"], errors="coerce")
            df = df.dropna()
            self.cache[indicator] = df
            logger.info(f"宏观指标 {indicator} 获取完成")
            return df
        except Exception as e:
            logger.warning(f"宏观指标 {indicator} 获取失败: {e}")
            return None

    def get_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """为数据添加宏观特征"""
        df = df.copy()
        if "date" not in df.columns:
            return df

        for indicator in self.MACRO_INDICATORS:
            macro_df = self.fetch_indicator(indicator)
            if macro_df is not None:
                df = df.merge(macro_df, on="date", how="left")

        df = df.fillna(method="ffill").fillna(0)
        logger.info("宏观特征添加完成")
        return df


class FeatureEngineer:
    """金融特征工程"""

    def __init__(self, config: dict[str, Any]):
        self.config = config.get("features", {})
        self.sentiment_generator = None
        self.macro_fetcher = None

        if self.config.get("sentiment_enabled", False):
            try:
                from src.data.sentiment_analyzer import SentimentFeatureGenerator
                self.sentiment_generator = SentimentFeatureGenerator(config)
                logger.info("情感分析模块已启用")
            except Exception as e:
                logger.warning(f"情感分析模块初始化失败（已禁用）: {e}")

        if self.config.get("macro_enabled", False):
            self.macro_fetcher = MacroIndicatorFetcher()
            logger.info("宏观指标模块已启用")

    def compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        tech_cfg = self.config.get("technical", {})
        ma_windows = tech_cfg.get("ma_windows", [5, 10, 20, 60])

        for w in ma_windows:
            df[f"ma_{w}"] = df["close"].rolling(window=w).mean()
            df[f"ma_{w}_bias"] = (df["close"] - df[f"ma_{w}"]) / df[f"ma_{w}"] * 100

        # RSI
        rsi_window = tech_cfg.get("rsi_window", 14)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
        rs = gain / loss.replace(0, 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        fast = tech_cfg.get("macd_fast", 12)
        slow = tech_cfg.get("macd_slow", 26)
        signal = tech_cfg.get("macd_signal", 9)
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # 布林带
        boll_window = tech_cfg.get("bollinger_window", 20)
        boll_std = tech_cfg.get("bollinger_std", 2)
        df["boll_mid"] = df["close"].rolling(window=boll_window).mean()
        df["boll_upper"] = df["boll_mid"] + boll_std * df["close"].rolling(window=boll_window).std()
        df["boll_lower"] = df["boll_mid"] - boll_std * df["close"].rolling(window=boll_window).std()
        df["boll_width"] = (df["boll_upper"] - df["boll_lower"]) / df["boll_mid"]

        logger.info("技术指标计算完成")
        return df

    def compute_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算收益率特征"""
        windows = self.config.get("returns", {}).get("windows", [1, 3, 5, 10, 20])
        for w in windows:
            df[f"return_{w}d"] = df["close"].pct_change(w) * 100
        logger.info("收益率特征计算完成")
        return df

    def compute_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算波动率特征"""
        windows = self.config.get("volatility", {}).get("windows", [5, 10, 20])
        for w in windows:
            df[f"volatility_{w}d"] = df["close"].pct_change().rolling(window=w).std() * np.sqrt(252) * 100
        logger.info("波动率特征计算完成")
        return df

    def compute_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算成交量特征"""
        ma_windows = self.config.get("volume", {}).get("ma_windows", [5, 10, 20])
        for w in ma_windows:
            df[f"volume_ma_{w}"] = df["volume"].rolling(window=w).mean()
            df[f"volume_ratio_{w}"] = df["volume"] / df[f"volume_ma_{w}"]
        logger.info("成交量特征计算完成")
        return df

    def create_target(self, df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
        """创建预测目标：未来N日收益率方向（1=涨，0=跌）"""
        df[f"future_return_{horizon}d"] = df["close"].shift(-horizon) / df["close"] - 1
        df[f"target_{horizon}d"] = (df[f"future_return_{horizon}d"] > 0).astype(int)
        logger.info(f"预测目标创建完成 (horizon={horizon}d)")
        return df

    def transform(self, df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
        """完整特征工程流水线"""
        df = df.copy()
        df = df.sort_values("date").reset_index(drop=True) if "date" in df.columns else df.reset_index(drop=True)

        # 确保数值列为 float 类型，避免 int64 运算出错
        for col in ["open", "close", "high", "low", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        df = self.compute_technical_indicators(df)
        df = self.compute_return_features(df)
        df = self.compute_volatility_features(df)
        df = self.compute_volume_features(df)

        # 宏观指标特征
        if self.macro_fetcher:
            df = self.macro_fetcher.get_macro_features(df)

        # 新闻情感特征
        if self.sentiment_generator:
            df = self.sentiment_generator.generate_sentiment_features(df)

        # 专业版扩展指标（50+ 指标）
        if self.config.get("extended_indicators", False):
            try:
                from src.data.indicators import TechnicalIndicators
                df = TechnicalIndicators.compute_all(df)
            except Exception as e:
                logger.warning(f"扩展指标计算失败（已跳过）: {e}")

        df = self.create_target(df, horizon)

        # 填充可选列为 0（腾讯数据源不提供 amount/turn）
        for col in ["amount", "turn"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        # 确保 date 为字符串（混合 Timestamp 和 str 会导致排序失败）
        if "date" in df.columns:
            df["date"] = df["date"].astype(str)

        # 删除 NaN 行
        before = len(df)
        df = df.dropna()
        after = len(df)
        logger.info(f"特征工程完成: {before} -> {after} 行 (删除 {before - after} 行 NaN)")
        return df

    def get_feature_columns(self, df: pd.DataFrame, horizon: int = 5) -> list[str]:
        """获取特征列名（排除目标和非特征列）"""
        exclude = {
            "date", "windcode", "close", "open", "high", "low", "volume", "amount", "turn", "pct_change",
            # 质量门控产生的非数值列
            "token_level_pred",
        }
        # 排除所有 future_return 和 target 列（无论 horizon 是多少）
        for col in df.columns:
            if col.startswith("future_return_") or col.startswith("target_"):
                exclude.add(col)
        return [c for c in df.columns if c not in exclude]


class DataPreprocessor:
    """数据预处理管道"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.processed_dir = Path(config["data"]["processed_dir"])
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.feature_engineer = FeatureEngineer(config)

    def process(self, all_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """处理所有标的数据"""
        horizon = self.config["data"]["forecast_horizon"]
        processed: dict[str, pd.DataFrame] = {}

        for symbol, df in all_data.items():
            logger.info(f"处理 {symbol}...")
            try:
                df_processed = self.feature_engineer.transform(df, horizon)
                save_path = self.processed_dir / f"{symbol}_processed.csv"
                df_processed.to_csv(save_path, index=False, encoding="utf-8-sig")
                processed[symbol] = df_processed
                logger.info(f"{symbol} 处理完成: {len(df_processed)} 行")
            except Exception as e:
                logger.error(f"处理 {symbol} 失败: {e}")

        return processed
