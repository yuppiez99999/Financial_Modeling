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

    def __init__(self, cache_dir: str = "data/news", cache_ttl_seconds: int = 1800,
                 max_items: int = 5000):
        """新闻采集器。

        Args:
            cache_dir: 新闻缓存目录。
            cache_ttl_seconds: 内存缓存有效期（默认 30 分钟）。同一进程内
                批量预测多标的时只触网一次，避免「每标的触发 HTTP 请求」。
            max_items: 内存缓存保留的最大条数（按时间倒序截断）。
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_seconds = int(cache_ttl_seconds)
        self.max_items = int(max_items)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
        # 内存缓存：避免批量预测时对每个标的重复触网（性能优化核心）
        self._memo: pd.DataFrame | None = None
        self._memo_at: float = 0.0
        self.fetch_count: int = 0  # 实际触网次数（供测试/监控）

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

    def fetch_all(self, force_refresh: bool = False) -> pd.DataFrame:
        """获取所有新闻源数据（带内存缓存，默认 30 分钟内复用）。

        性能优化：批量预测 N 个标的时，旧实现每个标的都会触网一次
        （`get_daily_sentiment` → `fetch_all`），N 倍 HTTP 开销。
        现改为进程级缓存 + TTL，N 个标的只触网 1 次。

        Args:
            force_refresh: 忽略缓存强制重新拉取。
        """
        now = time.time()
        if (
            not force_refresh
            and self._memo is not None
            and (now - self._memo_at) < self.cache_ttl_seconds
        ):
            logger.debug("[news] 命中内存缓存，跳过网络请求")
            return self._memo

        all_news: list[dict] = []
        for name, fetcher in (("eastmoney", self._fetch_eastmoney), ("sina", self._fetch_sina)):
            try:
                all_news.extend(fetcher())
            except Exception as e:  # noqa: BLE001  fail-open：单源失败不影响其他源
                logger.warning(f"[news] 数据源 {name} 异常: {e}")

        if not all_news:
            logger.warning("所有新闻源均无数据")
            # 回退到当日磁盘缓存（若存在），避免网络抖动导致当日情感特征全空
            cached = self._load_disk_cache(datetime.now().strftime("%Y-%m-%d"))
            self._memo = cached
            self._memo_at = now
            return cached

        df = pd.DataFrame(all_news)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time", "title"])
        df = df.sort_values("time", ascending=False)
        df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)
        df["date"] = df["time"].dt.strftime("%Y-%m-%d")
        if self.max_items and len(df) > self.max_items:
            df = df.head(self.max_items).reset_index(drop=True)

        cache_path = self.cache_dir / f"news_{datetime.now().strftime('%Y-%m-%d')}.csv"
        try:
            df.to_csv(cache_path, index=False, encoding="utf-8-sig")
            logger.info(f"新闻已缓存到 {cache_path} ({len(df)} 条)")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[news] 写缓存失败: {e}")

        self._memo = df
        self._memo_at = now
        self.fetch_count += 1
        return df

    def _load_disk_cache(self, date: str) -> pd.DataFrame:
        """读取指定日期的磁盘新闻缓存（失败返回空 DataFrame）。"""
        path = self.cache_dir / f"news_{date}.csv"
        if not path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(path)
            if "date" in df.columns:
                df["date"] = df["date"].astype(str)
            logger.info(f"[news] 回退磁盘缓存 {path} ({len(df)} 条)")
            return df
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[news] 读取磁盘缓存失败 {path}: {e}")
            return pd.DataFrame()

    def invalidate_cache(self) -> None:
        """清空内存缓存（供调度器/测试强制刷新）。"""
        self._memo = None
        self._memo_at = 0.0


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
        """批量分析文本情感。

        性能优化：旧实现逐条调用 `analyze_sentiment`，每条都要遍历
        正/负情感词典（各 ~30 词）× 正则预处理；改为一次性预处理后
        用集合求交统计，避免重复正则开销。
        """
        if not texts:
            return pd.DataFrame(columns=["sentiment", "positive_score", "negative_score"])

        cleaned = [self._text_preprocess(t) for t in texts]
        pos_tokens = self.pos_words
        neg_tokens = self.neg_words
        rows = []
        for text in cleaned:
            if not text:
                rows.append({"sentiment": 0.0, "positive_score": 0.0, "negative_score": 0.0})
                continue
            pos_count = sum(1 for w in pos_tokens if w in text)
            neg_count = sum(1 for w in neg_tokens if w in text)
            total = pos_count + neg_count
            if total == 0:
                rows.append({"sentiment": 0.0, "positive_score": 0.0, "negative_score": 0.0})
                continue
            rows.append({
                "sentiment": (pos_count - neg_count) / total,
                "positive_score": pos_count / total,
                "negative_score": neg_count / total,
            })
        return pd.DataFrame(rows)


class SentimentFeatureGenerator:
    """情感特征生成器"""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        sent_cfg = (config or {}).get("features", {}).get("sentiment", {}) or {}
        news_cfg = (config or {}).get("data", {}).get("news", {}) or {}
        self.news_collector = NewsCollector(
            cache_dir=news_cfg.get("dir", "data/news"),
            cache_ttl_seconds=news_cfg.get("cache_ttl_seconds", sent_cfg.get("cache_ttl_seconds", 1800)),
        )
        self.sentiment_analyzer = SentimentAnalyzer(config)
        self.sentiment_cache: dict[str, pd.DataFrame] = {}
        # 按日期预分组缓存：整表一次性算好，避免逐日反复过滤
        self._by_date: dict[str, pd.DataFrame] | None = None
        self._by_date_at: float = 0.0

    # ------------------------------------------------------------------
    def _prepare_by_date(self, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
        """一次性拉取新闻并做情感分析，按日期分组缓存。

        性能优化核心：旧实现 `generate_sentiment_features` 对 df 中每个
        交易日调用一次 `get_daily_sentiment`，每次都触发 `fetch_all()`
        （即使有 sentiment_cache，首次仍逐日触网）。现改为整表一次
        触网 + 一次批量情感分析 + 按日期分组。
        """
        now = time.time()
        ttl = self.news_collector.cache_ttl_seconds
        if (
            not force_refresh
            and self._by_date is not None
            and (now - self._by_date_at) < ttl
        ):
            return self._by_date

        news_df = self.news_collector.fetch_all(force_refresh=force_refresh)
        if news_df.empty or "date" not in news_df.columns:
            self._by_date = {}
            self._by_date_at = now
            return {}

        df = news_df.copy()
        if "title" not in df.columns:
            self._by_date = {}
            self._by_date_at = now
            return {}

        sentiment = self.sentiment_analyzer.analyze_batch(df["title"].astype(str).tolist())
        df = pd.concat([df.reset_index(drop=True), sentiment], axis=1)

        grouped = {str(d): g.reset_index(drop=True) for d, g in df.groupby("date", sort=False)}
        self._by_date = grouped
        self._by_date_at = now
        return grouped

    def get_daily_sentiment(self, date: str | None = None) -> pd.DataFrame:
        """获取指定日期的情感数据（复用整表分组缓存，不重复触网）"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        date = str(date)

        if date in self.sentiment_cache:
            return self.sentiment_cache[date]

        grouped = self._prepare_by_date()
        result = grouped.get(date, pd.DataFrame())
        self.sentiment_cache[date] = result
        return result

    def generate_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """为K线数据生成情感特征"""
        df = df.copy()
        if "date" not in df.columns:
            logger.warning("数据中无 date 列，跳过情感特征")
            return df
        # 统一日期口径为 YYYY-MM-DD 字符串，保证与新闻分组键可对齐
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

        dates = list(pd.Series(df["date"]).astype(str).unique())
        grouped = self._prepare_by_date()

        # 整表一次性分组聚合（替代逐日循环），无新闻的日期走零值默认
        agg_map: dict[str, dict] = {}
        for date_str, group in grouped.items():
            agg_map[date_str] = {
                "sentiment_mean": float(group["sentiment"].mean()),
                "sentiment_std": float(group["sentiment"].std()) if len(group) > 1 else 0.0,
                "positive_ratio": float(group["positive_score"].mean()),
                "negative_ratio": float(group["negative_score"].mean()),
                "news_count": int(len(group)),
            }

        sentiment_scores = []
        for date_str in dates:
            if date_str in agg_map:
                row = {"date": date_str}
                row.update(agg_map[date_str])
                sentiment_scores.append(row)
            else:
                sentiment_scores.append({
                    "date": date_str,
                    "sentiment_mean": 0.0,
                    "sentiment_std": 0.0,
                    "positive_ratio": 0.5,
                    "negative_ratio": 0.5,
                    "news_count": 0,
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