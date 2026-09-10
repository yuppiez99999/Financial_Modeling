"""腾讯数据源客户端 + 采集器/预处理器回归测试（全 mock，不触网）。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.collector import DataCollector  # noqa: E402
from src.data.preprocessor import DataPreprocessor, FeatureEngineer  # noqa: E402
from src.data.tencent_client import TencentClient, to_tencent_code  # noqa: E402


# ---------- 代码映射 ----------
def test_to_tencent_code():
    assert to_tencent_code("600519.SH") == "sh600519"
    assert to_tencent_code("000858.SZ") == "sz000858"
    assert to_tencent_code("510300.SH") == "sh510300"
    assert to_tencent_code("159915.SZ") == "sz159915"
    assert to_tencent_code("rb.shf") is None       # 期货不支持（大小写不敏感）
    assert to_tencent_code("USDCNH.FXCM") is None   # 外汇不支持
    assert to_tencent_code("") is None


# ---------- 腾讯原始行解析（列序重排是关键口径） ----------
def test_parse_rows_reorders_tencent_columns():
    # 腾讯原始列序: date, open, close, high, low, volume
    rows = [
        ["2026-09-08", "1318.000", "1309.300", "1323.000", "1309.050", "17534.000"],
        ["2026-09-09", "1305.01", "1290.88", "1309.30", "1286.68", "32226"],
    ]
    df = TencentClient._parse_rows(rows)
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
    row0 = df.iloc[0]
    assert row0["open"] == 1318.000
    assert row0["close"] == 1309.300
    assert row0["high"] == 1323.000
    assert row0["low"] == 1309.050
    assert row0["volume"] == 17534.000
    assert df["date"].iloc[0] == "2026-09-08"


# ---------- fetch：成功 / 异常 / 不支持 / 空数据 / 分页 ----------
class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _payload(rows, code="sh600519"):
    return {"code": 0, "data": {code: {"qfqday": rows}}}


def test_fetch_success(monkeypatch):
    client = TencentClient({})
    rows = [
        ["2026-09-08", "10", "11", "12", "9.5", "100"],
        ["2026-09-09", "11", "10.5", "11.8", "10.2", "200"],
    ]
    monkeypatch.setattr(client.session, "get", lambda *a, **k: _FakeResp(_payload(rows)))
    df = client.fetch("600519.SH")
    assert df is not None and len(df) == 2
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert df["close"].iloc[-1] == 10.5


def test_fetch_network_error_returns_none(monkeypatch):
    client = TencentClient({})

    def _boom(*a, **k):
        raise ConnectionError("network down")

    monkeypatch.setattr(client.session, "get", _boom)
    assert client.fetch("600519.SH") is None  # fail-open


def test_fetch_unsupported_symbol():
    assert TencentClient({}).fetch("RB.SHF") is None


def test_fetch_empty_data_returns_none(monkeypatch):
    client = TencentClient({})
    monkeypatch.setattr(
        client.session, "get", lambda *a, **k: _FakeResp({"code": 0, "data": {}})
    )
    assert client.fetch("600519.SH") is None


def test_fetch_day_key_fallback(monkeypatch):
    """qfq 不可用时接口返回 day 键，同样要能解析。"""
    client = TencentClient({})
    payload = {
        "code": 0,
        "data": {"sz000858": {"day": [["2026-09-09", "9", "10", "11", "8", "50"]]}},
    }
    monkeypatch.setattr(client.session, "get", lambda *a, **k: _FakeResp(payload))
    df = client.fetch("000858.SZ")
    assert df is not None and len(df) == 1
    assert df["close"].iloc[0] == 10.0


def test_fetch_pagination_until_start(monkeypatch):
    client = TencentClient({})
    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            rows = [["2026-02-03", "10", "11", "12", "9", "100"]] * 3
        else:
            rows = [["2025-12-01", "9", "10", "11", "8", "100"]]
        return _FakeResp(_payload(rows))

    monkeypatch.setattr(client.session, "get", fake_get)
    df = client.fetch("600519.SH", start_date="2026-01-01")
    assert calls["n"] == 2  # 首页最早 2026-02-03 > start → 向历史回翻一页
    assert df is not None and len(df) == 4
    assert df["date"].iloc[0] == "2025-12-01"  # 历史页拼在前面（保持升序）


# ---------- DataCollector 集成（mock，不触网） ----------
def _cfg(tmp_path, sources=("tencent",)):
    return {
        "training": {"save_dir": str(tmp_path / "models")},
        "data": {
            "source": list(sources),
            "raw_dir": str(tmp_path / "raw"),
            "start_date": "2026-01-01",
            "end_date": "2026-03-01",
            "markets": {
                "stock": {"enabled": True, "symbols": ["600519.SH", "000858.SZ"]},
                "futures": {"enabled": False, "symbols": ["RB.SHF"]},
            },
        },
    }


def _fake_ohlcv(symbol, n=300):
    """2025-06-01 起 300 个日历日，覆盖测试窗口 [2026-01-01, 2026-03-01]。"""
    dates = pd.date_range("2025-06-01", periods=n, freq="D").strftime("%Y-%m-%d")
    close = 100 + np.arange(n) * 0.1
    return pd.DataFrame({
        "date": dates, "open": close, "high": close * 1.01,
        "low": close * 0.99, "close": close, "volume": np.arange(n) * 10.0,
    })


def test_fetch_with_fallback_tencent_saves_cache(tmp_path, monkeypatch):
    col = DataCollector(_cfg(tmp_path))
    monkeypatch.setattr(col, "_fetch_tencent", lambda s, **k: _fake_ohlcv(s))
    df = col._fetch_with_fallback("600519.SH")
    assert df is not None and len(df) == 300
    assert (tmp_path / "raw" / "600519.SH.csv").exists()


def test_fetch_with_fallback_simulation_no_cache(tmp_path):
    """simulation 兜底数据禁止落盘（防污染 data/raw 缓存被推理误用）。"""
    col = DataCollector(_cfg(tmp_path, sources=("simulation",)))
    df = col._fetch_with_fallback("TEST.SZ")
    assert df is not None and len(df) > 0
    assert not (tmp_path / "raw" / "TEST.SZ.csv").exists()


def test_load_cached_rejects_missing_columns(tmp_path):
    col = DataCollector(_cfg(tmp_path))
    bad = pd.DataFrame({"date": ["2026-01-02"], "close": [10.0]})
    bad.to_csv(tmp_path / "raw" / "BAD.SZ.csv", index=False)
    assert col.load_cached("BAD.SZ") is None
    assert col.load_cached("NOTEXIST.SZ") is None


def test_collect_all_window_and_skip_disabled(tmp_path, monkeypatch):
    col = DataCollector(_cfg(tmp_path))
    seen = {}

    def fake_fetch(symbol, start_date="", end_date=""):
        seen[symbol] = (start_date, end_date)
        return _fake_ohlcv(symbol)

    monkeypatch.setattr(col, "_fetch_with_fallback", fake_fetch)
    all_data = col.collect_all()
    # futures enabled=false → 不采集；只采两只股票
    assert set(all_data.keys()) == {"600519.SH", "000858.SZ"}
    # 窗口参数透传 + 结果按窗口截取（300行 → 窗口内 60 行）
    assert seen["600519.SH"] == ("2026-01-01", "2026-03-01")
    assert all(len(df) == 60 for df in all_data.values())


def test_collect_all_skips_failed_symbol(tmp_path, monkeypatch):
    col = DataCollector(_cfg(tmp_path))
    monkeypatch.setattr(col, "_fetch_with_fallback", lambda s, **k: None)
    assert col.collect_all() == {}


# ---------- 特征工程 / 预处理器（防目标泄漏回归） ----------
def _sample_raw(n=80, base=100.0):
    rng = np.random.default_rng(3)
    close = base * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "date": dates, "open": close, "high": close * 1.01,
        "low": close * 0.99, "close": close,
        "volume": rng.integers(1_000, 100_000, n).astype(float),
    })


def test_create_target_values():
    fe = FeatureEngineer({})
    df = pd.DataFrame({"close": [100.0, 110.0, 90.0, 105.0, 95.0, 120.0]})
    out = fe.create_target(df, 1)
    # 未来1日收益: +10%→1, -18%→0, +16%→1, -9.5%→0, +26%→1
    assert out["target_1d"].tolist()[:5] == [1.0, 0.0, 1.0, 0.0, 1.0]
    # 尾行无未来数据 → NaN（不能用 == 比较 nan）
    assert bool(np.isnan(out["target_1d"].iloc[-1]))


def test_feature_columns_exclude_target_no_leakage():
    fe = FeatureEngineer({})
    feats = fe.create_target(fe.transform(_sample_raw(), 5), 5)
    cols = fe.get_feature_columns(feats, 5)
    assert not any(str(c).startswith("target_") for c in cols)
    assert "rsi" in cols and "macd" in cols
    assert not {"date", "open", "high", "low", "close", "volume"} & set(cols)


def test_data_preprocessor_process():
    pre = DataPreprocessor({"features": {"technical": {"ma_windows": [5, 10]}}})
    out = pre.process({"AAA": _sample_raw(), "BBB": _sample_raw()})
    assert set(out.keys()) == {"AAA", "BBB"}
    assert "ma_5" in out["AAA"].columns
    assert "target_5d" not in out["AAA"].columns  # process 不构造目标