"""Q1 排期：新闻采集性能优化测试。

核心验收：批量预测 N 个标的时，新闻源只触网 1 次（旧实现 N 次）。
全部离线运行（mock 网络）。
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.sentiment_analyzer import (
    NewsCollector,
    SentimentAnalyzer,
    SentimentFeatureGenerator,
)


@pytest.fixture()
def fake_news(monkeypatch):
    """mock 两个新闻源，并统计实际调用次数。"""
    calls = {"eastmoney": 0, "sina": 0}

    def fake_eastmoney(self):
        calls["eastmoney"] += 1
        return [
            {"source": "eastmoney", "title": "公司业绩大涨 超预期", "time": "2024-03-01 09:00:00"},
            {"source": "eastmoney", "title": "行业风险加剧 亏损扩大", "time": "2024-03-02 10:00:00"},
        ]

    def fake_sina(self):
        calls["sina"] += 1
        return [
            {"source": "sina", "title": "政策利好 市场回暖上涨", "time": "2024-03-01 11:00:00"},
        ]

    monkeypatch.setattr(NewsCollector, "_fetch_eastmoney", fake_eastmoney)
    monkeypatch.setattr(NewsCollector, "_fetch_sina", fake_sina)
    return calls


# ---------- 采集缓存 ----------

def test_fetch_all_caches_within_ttl(tmp_path, fake_news):
    collector = NewsCollector(cache_dir=str(tmp_path), cache_ttl_seconds=1800)
    first = collector.fetch_all()
    second = collector.fetch_all()
    assert len(first) == 3
    assert collector.fetch_count == 1        # 只触网一次
    assert fake_news["eastmoney"] == 1
    assert fake_news["sina"] == 1
    assert first.equals(second)


def test_fetch_all_force_refresh(tmp_path, fake_news):
    collector = NewsCollector(cache_dir=str(tmp_path), cache_ttl_seconds=1800)
    collector.fetch_all()
    collector.fetch_all(force_refresh=True)
    assert collector.fetch_count == 2
    assert fake_news["eastmoney"] == 2


def test_fetch_all_ttl_expiry(tmp_path, fake_news):
    collector = NewsCollector(cache_dir=str(tmp_path), cache_ttl_seconds=0)
    collector.fetch_all()
    collector.fetch_all()
    assert collector.fetch_count == 2


def test_fetch_all_dedup_and_sort(tmp_path, monkeypatch):
    def dup_east(self):
        return [
            {"source": "e", "title": "重复标题", "time": "2024-03-01 09:00:00"},
            {"source": "e", "title": "重复标题", "time": "2024-03-01 10:00:00"},
            {"source": "e", "title": "另一条", "time": "2024-03-03 10:00:00"},
        ]

    monkeypatch.setattr(NewsCollector, "_fetch_eastmoney", dup_east)
    monkeypatch.setattr(NewsCollector, "_fetch_sina", lambda self: [])
    df = NewsCollector(cache_dir=str(tmp_path)).fetch_all()
    assert len(df) == 2                       # 去重
    assert df["time"].is_monotonic_decreasing  # 时间倒序


def test_fetch_all_falls_back_to_disk_cache(tmp_path, fake_news):
    collector = NewsCollector(cache_dir=str(tmp_path))
    collector.fetch_all()                     # 写入磁盘缓存
    collector.invalidate_cache()

    monkeypatch_off = lambda self: []         # noqa: E731
    NewsCollector._fetch_eastmoney = monkeypatch_off
    NewsCollector._fetch_sina = monkeypatch_off
    fresh = NewsCollector(cache_dir=str(tmp_path))
    df = fresh.fetch_all()
    assert len(df) == 3                       # 网络全挂 → 回退磁盘缓存


def test_fetch_all_source_failure_is_fail_open(tmp_path, monkeypatch):
    def boom(self):
        raise RuntimeError("network down")

    monkeypatch.setattr(NewsCollector, "_fetch_eastmoney", boom)
    monkeypatch.setattr(NewsCollector, "_fetch_sina", lambda self: [
        {"source": "sina", "title": "上涨", "time": "2024-03-01 09:00:00"},
    ])
    df = NewsCollector(cache_dir=str(tmp_path)).fetch_all()
    assert len(df) == 1


def test_max_items_truncates(tmp_path, monkeypatch):
    def many(self):
        return [
            {"source": "e", "title": f"新闻{i}", "time": f"2024-03-{i % 28 + 1:02d} 09:00:00"}
            for i in range(50)
        ]

    monkeypatch.setattr(NewsCollector, "_fetch_eastmoney", many)
    monkeypatch.setattr(NewsCollector, "_fetch_sina", lambda self: [])
    df = NewsCollector(cache_dir=str(tmp_path), max_items=10).fetch_all()
    assert len(df) == 10


# ---------- 批量情感分析 ----------

def test_analyze_batch_matches_single():
    analyzer = SentimentAnalyzer()
    texts = ["业绩大涨 超预期", "暴跌 亏损 风险", "平平无奇", ""]
    batch = analyzer.analyze_batch(texts)
    assert len(batch) == len(texts)
    for i, t in enumerate(texts):
        single = analyzer.analyze_sentiment(t)
        assert batch["sentiment"].iloc[i] == pytest.approx(single["sentiment"])
        assert batch["positive_score"].iloc[i] == pytest.approx(single["positive_score"])


def test_analyze_batch_empty():
    df = SentimentAnalyzer().analyze_batch([])
    assert df.empty and list(df.columns) == ["sentiment", "positive_score", "negative_score"]


# ---------- 端到端：N 个交易日只触网 1 次 ----------

def test_generate_features_fetches_news_once(tmp_path, fake_news):
    """批量预测场景：多日/多标的情感特征生成只触发 1 次新闻采集。"""
    gen = SentimentFeatureGenerator({
        "data": {"news": {"dir": str(tmp_path), "cache_ttl_seconds": 1800}},
    })
    dates = pd.date_range("2024-03-01", periods=10)
    df = pd.DataFrame({"date": dates, "close": range(10)})

    out = gen.generate_sentiment_features(df)
    assert len(out) == 10
    assert gen.news_collector.fetch_count == 1      # 关键：只触网 1 次
    assert fake_news["eastmoney"] == 1
    assert fake_news["sina"] == 1


def test_generate_features_reuses_across_symbols(tmp_path, fake_news):
    """多标的复用同一 generator：总触网次数仍为 1。"""
    gen = SentimentFeatureGenerator({"data": {"news": {"dir": str(tmp_path)}}})
    df = pd.DataFrame({"date": pd.date_range("2024-03-01", periods=5), "close": range(5)})
    for _ in range(20):                              # 模拟 20 个标的
        gen.generate_sentiment_features(df)
    assert gen.news_collector.fetch_count == 1


def test_generate_features_values_and_defaults(tmp_path, fake_news):
    gen = SentimentFeatureGenerator({"data": {"news": {"dir": str(tmp_path)}}})
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-03-01", "2024-03-09"]),  # 后者无新闻
        "close": [1.0, 2.0],
    })
    out = gen.generate_sentiment_features(df)
    row0, row1 = out.iloc[0], out.iloc[1]
    assert row0["news_count"] == 2
    assert row0["sentiment_mean"] > 0                # 2 正 1 负
    assert row1["news_count"] == 0
    assert row1["positive_ratio"] == 0.5 and row1["negative_ratio"] == 0.5


def test_generate_features_missing_date_column(tmp_path, fake_news):
    gen = SentimentFeatureGenerator({"data": {"news": {"dir": str(tmp_path)}}})
    df = pd.DataFrame({"close": [1.0, 2.0]})
    out = gen.generate_sentiment_features(df)
    assert list(out.columns) == ["close"]
    assert gen.news_collector.fetch_count == 0


def test_generate_features_handles_empty_news(tmp_path, monkeypatch):
    monkeypatch.setattr(NewsCollector, "_fetch_eastmoney", lambda self: [])
    monkeypatch.setattr(NewsCollector, "_fetch_sina", lambda self: [])
    gen = SentimentFeatureGenerator({"data": {"news": {"dir": str(tmp_path)}}})
    df = pd.DataFrame({"date": pd.date_range("2024-03-01", periods=3), "close": [1.0, 2.0, 3.0]})
    out = gen.generate_sentiment_features(df)
    assert (out["news_count"] == 0).all()
    assert (out["sentiment_mean"] == 0.0).all()
