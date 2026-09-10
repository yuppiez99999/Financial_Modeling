"""推理数据每日刷新（_ensure_fresh）单元测试（全 mock，不触网）。"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import PredictionEngine  # noqa: E402


def _cfg(tmp_path):
    return {
        "training": {"save_dir": str(tmp_path / "models")},
        "data": {"prediction_horizons": {"short_term": 5}},
        "model": {"type": "lightgbm"},
    }


def _df_ending(days_ago: int, n: int = 30):
    """最后日期为 days_ago 天前的行情 DataFrame。"""
    end = pd.Timestamp.now().normalize() - pd.Timedelta(days=days_ago)
    dates = pd.date_range(end=end, periods=n)
    close = 100.0 + pd.Series(range(n), dtype=float)
    return pd.DataFrame({
        "date": dates, "open": close, "high": close * 1.01,
        "low": close * 0.99, "close": close, "volume": 1000.0,
    })


class FakeCollector:
    """记录 fetch_realtime 调用并按脚本返回。"""

    def __init__(self, results):
        self.results = list(results)  # 每次调用弹出一个返回值（缺省 None）
        self.calls = []

    def fetch_realtime(self, symbol, start_date="", end_date=""):
        self.calls.append(symbol)
        return self.results.pop(0) if self.results else None


def _engine(tmp_path):
    return PredictionEngine(_cfg(tmp_path))


def test_fresh_cache_skips_refresh(tmp_path):
    eng = _engine(tmp_path)
    col = FakeCollector(results=[])
    fresh = _df_ending(0)
    out = eng._ensure_fresh("600519.SH", fresh, col)
    assert out is fresh
    assert col.calls == []  # 缓存最新 → 不触发网络


def test_stale_cache_refreshes_and_adopts(tmp_path):
    eng = _engine(tmp_path)
    newer = _df_ending(0)
    col = FakeCollector(results=[newer])
    stale = _df_ending(5)
    out = eng._ensure_fresh("600519.SH", stale, col)
    assert out is newer
    assert col.calls == ["600519.SH"]


def test_stale_cache_fetch_none_falls_back(tmp_path):
    eng = _engine(tmp_path)
    col = FakeCollector(results=[None])
    stale = _df_ending(5)
    out = eng._ensure_fresh("600519.SH", stale, col)
    assert out is stale  # fail-open 回退旧缓存


def test_stale_cache_no_newer_data_falls_back_once(tmp_path):
    eng = _engine(tmp_path)
    same = _df_ending(5)  # 腾讯在非交易日返回的数据不比缓存新
    col = FakeCollector(results=[same])
    stale = _df_ending(5)
    out1 = eng._ensure_fresh("600519.SH", stale, col)
    assert out1 is stale
    # 同进程第二次预测（同标的）不再重复刷新
    out2 = eng._ensure_fresh("600519.SH", stale, col)
    assert out2 is stale
    assert col.calls == ["600519.SH"]  # 仅一次调用


def test_no_cache_triggers_refresh(tmp_path):
    eng = _engine(tmp_path)
    fresh = _df_ending(0)
    col = FakeCollector(results=[fresh])
    out = eng._ensure_fresh("NEW.SH", None, col)
    assert out is fresh
    assert col.calls == ["NEW.SH"]


def test_simulation_like_result_older_than_today_is_still_used_only_if_newer(tmp_path):
    """刷新结果若日期不前进则不被采用（防 simulation 假数据伪装新鲜）。"""
    eng = _engine(tmp_path)
    real_stale = _df_ending(3)
    sim_like = _df_ending(3)  # 假数据：与真实缓存同最后日期
    col = FakeCollector(results=[sim_like])
    out = eng._ensure_fresh("600276.SH", real_stale, col)
    assert out is real_stale
