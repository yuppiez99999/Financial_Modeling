"""金融市场预测模型 - 新闻情感分析模块

支持多源新闻采集和情感分析，为预测模型提供市场情绪特征。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


class NewsCollector:
    """多源财经新闻采集器"""

    NEWS_SOURCES = {
        "eastmoney": {
            "name": "东方财富",
            "url": "https://push2.eastmoney.com/api/qt/clist/get",
            "params": {
                "pn": "1",
                "pz": "50",
                "po": "1",
                "np": "1",
                "ut": "bd1d9ddb04089700cf9c27f6f7426281",
                "fltt": "2",
                "invt": "2",
                "fid": "f3",
                "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23",
                "fields": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f26,f22,f33,f11,f62,f128,f136,f115,f152",
            },
            "headers": {"User-Agent": "Mozilla/5.0"},
        },
        "sina": {
            "name": "新浪财经",
            "url": "https://feed.mix.sina.com.cn/api/roll/get",
            "params": {
                "pageid": "153",
                "lid": "2509",
                "k": "",
                "num": "50",
                "page": "1",
            },
            "headers": {"User-Agent": "Mozilla/5.0"},
        },
    }

    def __init__(self, cache_dir: str = "data/news"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})

    def _fetch_eastmoney(self) -> list[dict]:
        """获取东方财富新闻"""
        try:
            cfg = self.NEWS_SOURCES["eastmoney"]
            resp = self._session.get(cfg["url"], params=cfg["params"], timeout=15)
            data = resp.json()
            news_list = []
            for item in data.get("data", {}).get("diff", []):
                news_list.append({
                    "source": "eastmoney",
                    "title": item.get("f14", ""),
                    "time": item.get("f15", ""),
                    "code": item.get("f12", ""),
                    "name": item.get("f14", ""),
                })
            logger.info(f"东方财富: 获取 {len(news_list)} 条新闻")
            return news_list
        except Exception as e:
            logger.warning(f"东方财富新闻获取失败: {e}")
            return []

    def _fetch_sina(self) -> list[dict]:
        """获取新浪财经新闻"""
        try:
            cfg = self.NEWS_SOURCES["sina"]
            resp = self._session.get(cfg["url"], params=cfg["params"], timeout=15)
            data = resp.json()
            news_list = []
            for item in data.get("result", {}).get("data", []):
                news_list.append({
                    "source": "sina",
                    "title": item.get("title", ""),
                    "time": item.get("ctime", ""),
                    "url": item.get("url", ""),
                })
            logger.info(f"新浪财经: 获取 {len(news_list)} 条新闻")
            return news_list
        except Exception as e:
            logger.warning(f"新浪财经新闻获取失败: {e}")
            return []

    def fetch_all(self) -> pd.DataFrame:
        """获取所有新闻源数据"""
        all_news = []
        all_news.extend(self._fetch_eastmoney())
        all_news.extend(self._fetch_sina())

        if not all_news:
            logger.warning("所有新闻源均无数据")
            return pd.DataFrame()

        df = pd.DataFrame(all_news)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time", "title"])
        df = df.sort_values("time", ascending=False)
        df["date"] = df["time"].dt.strftime("%Y-%m-%d")

        cache_path = self.cache_dir / f"news_{datetime.now().strftime('%Y-%m-%d')}.csv"
        df.to_csv(cache_path, index=False, encoding="utf-8-sig")
        logger.info(f"新闻已缓存到 {cache_path} ({len(df)} 条)")
        return df


class SentimentAnalyzer:
    """新闻情感分析器"""

    POSITIVE_WORDS = {
        "上涨", "大涨", "暴涨", "飙升", "涨停", "利好", "增持", "买入",
        "突破", "创新高", "走强", "反弹", "回暖", "利好", "看好", "强劲",
        "增长", "盈利", "超预期", "业绩", "优秀", "强势", "利好", "升",
        "增", "涨", "高", "好", "优", "强", "多", "赢", "胜",
    }

    NEGATIVE_WORDS = {
        "下跌", "大跌", "暴跌", "跳水", "跌停", "利空", "减持", "卖出",
        "跌破", "创新低", "走弱", "回落", "低迷", "利空", "看空", "疲软",
        "下滑", "亏损", "不及预期", "爆雷", "风险", "负面", "利空", "降",
        "减", "跌", "低", "差", "弱", "空", "输", "亏",
    }

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.pos_words = self.POSITIVE_WORDS
        self.neg_words = self.NEGATIVE_WORDS

    def _text_preprocess(self, text: str) -> str:
        """文本预处理"""
        text = str(text).strip()
        text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def analyze_sentiment(self, text: str) -> dict[str, float]:
        """分析单条文本情感"""
        text = self._text_preprocess(text)
        if not text:
            return {"sentiment": 0.0, "positive_score": 0.0, "negative_score": 0.0}

        pos_count = sum(1 for w in self.pos_words if w in text)
        neg_count = sum(1 for w in self.neg_words if w in text)
        total = pos_count + neg_count

        if total == 0:
            return {"sentiment": 0.0, "positive_score": 0.0, "negative_score": 0.0}

        sentiment = (pos_count - neg_count) / total
        positive_score = pos_count / total
        negative_score = neg_count / total

        return {
            "sentiment": sentiment,
            "positive_score": positive_score,
            "negative_score": negative_score,
        }

    def analyze_batch(self, texts: list[str]) -> pd.DataFrame:
        """批量分析文本情感"""
        results = []
        for text in texts:
            result = self.analyze_sentiment(text)
            results.append(result)
        return pd.DataFrame(results)


class SentimentFeatureGenerator:
    """情感特征生成器"""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.news_collector = NewsCollector()
        self.sentiment_analyzer = SentimentAnalyzer(config)
        self.sentiment_cache: dict[str, pd.DataFrame] = {}

    def get_daily_sentiment(self, date: str | None = None) -> pd.DataFrame:
        """获取指定日期的情感数据"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        if date in self.sentiment_cache:
            return self.sentiment_cache[date]

        news_df = self.news_collector.fetch_all()
        if news_df.empty:
            return pd.DataFrame()

        filtered = news_df[news_df["date"] == date]
        if filtered.empty:
            return pd.DataFrame()

        sentiment_results = self.sentiment_analyzer.analyze_batch(filtered["title"].tolist())
        result = pd.concat([filtered.reset_index(drop=True), sentiment_results], axis=1)
        self.sentiment_cache[date] = result
        return result

    def generate_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """为K线数据生成情感特征"""
        df = df.copy()
        if "date" not in df.columns:
            logger.warning("数据中无 date 列，跳过情感特征")
            return df

        dates = df["date"].unique()
        sentiment_scores = []

        for date_str in dates:
            sentiment_df = self.get_daily_sentiment(date_str)
            if sentiment_df.empty:
                sentiment_scores.append({
                    "date": date_str,
                    "sentiment_mean": 0.0,
                    "sentiment_std": 0.0,
                    "positive_ratio": 0.5,
                    "negative_ratio": 0.5,
                    "news_count": 0,
                })
            else:
                sentiment_scores.append({
                    "date": date_str,
                    "sentiment_mean": float(sentiment_df["sentiment"].mean()),
                    "sentiment_std": float(sentiment_df["sentiment"].std()),
                    "positive_ratio": float(sentiment_df["positive_score"].mean()),
                    "negative_ratio": float(sentiment_df["negative_score"].mean()),
                    "news_count": int(len(sentiment_df)),
                })

        sentiment_df = pd.DataFrame(sentiment_scores)
        df = df.merge(sentiment_df, on="date", how="left")
        df = df.fillna({
            "sentiment_mean": 0.0,
            "sentiment_std": 0.0,
            "positive_ratio": 0.5,
            "negative_ratio": 0.5,
            "news_count": 0,
        })

        logger.info("情感特征生成完成")
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    analyzer = SentimentFeatureGenerator()
    news = analyzer.news_collector.fetch_all()
    print(f"获取新闻: {len(news)} 条")
    if not news.empty:
        sentiment = analyzer.sentiment_analyzer.analyze_batch(news["title"].head(10).tolist())
        print("\n情感分析示例:")
        for i, (title, sent) in enumerate(zip(news["title"].head(10), sentiment["sentiment"])):
            print(f"{i+1}. {title[:50]}... -> 情感分: {sent:.2f}")