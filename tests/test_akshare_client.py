"""S14（G4）akshare 数据源测试：代码映射 / 多市场取数 / 回退链 / 离线降级。

测试要点（对照 README §18A 统一原则）：
  - 代码映射：A股 / ETF / 国内期货 / 外汇 / 未知代码，市场判定按代码形态而非字典；
  - 列序归一：**新浪外汇日K 的 date/open/low/high/close 列序** 是交叉验证结论，
    必须固定（取错列 = 静默污染全部外汇特征）；
  - 退化行剔除：非正价格 / high<low 一律剔除，与腾讯源同口径；
  - volume 占位：外汇无量 → 如实填 1.0（防下游量类特征除零），**不编造价格**；
  - 回退链：DataCollector 中 akshare 位于 wind 之后、tencent 之前；akshare 成功
    即短路，失败继续降级（fail-open）；
  - 未安装 akshare / 未识别代码 → 返回 None 而非抛异常；
  - 窗口裁剪按 start/end 生效。

全部离线（akshare 接口调用一律注入 fake，不触网）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.akshare_client import (  # noqa: E402
    AkshareClient,
    STD_COLUMNS,
    akshare_available,
    split_symbol,
    to_futures_sina_code,
    to_fx_sina_symbol,
    to_sina_ashare_code,
)
from src.data.collector import DataCollector  # noqa: E402


def _raw(n: int = 60, seed: int = 3, cols=None) -> pd.DataFrame:
    """构造指定列序的原始行情（模拟 akshare 返回值）。"""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    data = {
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": close * 1.001,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": rng.integers(1_000, 10_000, n).astype(float),
    }
    extras = {
        "amount": close * 1e6,
        "hold": rng.integers(1_000, 9_000, n).astype(float),
        "settle": close * 0.999,
        "turnover": rng.random(n) * 0.01,
    }
    data.update({k: v for k, v in extras.items() if k not in data})
    cols = cols or ["date", "open", "high", "low", "close", "volume"]
    return pd.DataFrame({c: data[c] for c in cols})


# ----------------------------------------------------------------------
# 代码映射
# ----------------------------------------------------------------------
class TestSymbolMapping:
    def test_split_symbol_markets(self):
        assert split_symbol("600519.SH") == ("stock", "600519")
        assert split_symbol("000858.SZ") == ("stock", "000858")
        assert split_symbol("300308.SZ") == ("stock", "300308")
        # 场内基金 / ETF：5xx 沪、15x/16x/18x 深
        assert split_symbol("510300.SH") == ("etf", "510300")
        assert split_symbol("159915.SZ") == ("etf", "159915")
        assert split_symbol("588080.SH") == ("etf", "588080")
        # 期货：交易所后缀
        assert split_symbol("RB.SHF") == ("futures", "RB")
        assert split_symbol("SC.INE") == ("futures", "SC")
        assert split_symbol("I.DCE") == ("futures", "I")
        assert split_symbol("SR.CZC") == ("futures", "SR")
        # 外汇
        assert split_symbol("USDCNH.FXCM") == ("forex", "USDCNH")
        # 未知
        assert split_symbol("") == ("unknown", "")
        assert split_symbol("XXX") == ("unknown", "XXX")
        assert split_symbol("ABC.QQ") == ("unknown", "ABC")

    def test_split_symbol_case_insensitive(self):
        assert split_symbol("rb.shf") == ("futures", "RB")
        assert split_symbol("usdcnh.fxcm") == ("forex", "USDCNH")

    def test_to_sina_ashare_code(self):
        assert to_sina_ashare_code("600519.SH") == "sh600519"
        assert to_sina_ashare_code("159915.SZ") == "sz159915"
        assert to_sina_ashare_code("RB.SHF") is None       # 期货不走 A股通道
        assert to_sina_ashare_code("USDCNH.FXCM") is None  # 外汇不走 A股通道

    def test_to_futures_sina_code(self):
        assert to_futures_sina_code("RB.SHF") == "RB0"
        assert to_futures_sina_code("SC.INE") == "SC0"
        assert to_futures_sina_code("600519.SH") is None

    def test_to_fx_sina_symbol(self):
        assert to_fx_sina_symbol("USDCNH.FXCM") == "fx_susdcnh"
        assert to_fx_sina_symbol("USDCNY.FXCM") == "fx_susdcny"
        assert to_fx_sina_symbol("RB.SHF") is None
        # 未登记的外汇代码：如实返回 None（不猜价格，也不猜 symbol）
        assert to_fx_sina_symbol("ZZZ.FXCM") is None


# ----------------------------------------------------------------------
# 列序归一（关键口径）
# ----------------------------------------------------------------------
class TestNormalize:
    def test_sina_fx_column_order_is_fixed(self):
        """新浪外汇日K列序 = date/open/low/high/close，取错列会静默污染全部外汇特征。"""
        raw = pd.DataFrame({
            "date": ["2026-09-09", "2026-09-10"],
            "open": ["6.70660", "6.70640"],
            "low": ["6.70220", "6.70370"],
            "high": ["6.70770", "6.71500"],
            "close": ["6.70640", "6.71450"],
        })
        out = AkshareClient._normalize(raw, "USDCNH.FXCM")
        assert list(out.columns) == STD_COLUMNS
        row = out.iloc[-1]
        assert row["open"] == pytest.approx(6.70640)
        assert row["low"] == pytest.approx(6.70370)
        assert row["high"] == pytest.approx(6.71500)
        assert row["close"] == pytest.approx(6.71450)
        assert row["low"] <= min(row["open"], row["close"]) <= max(row["open"], row["close"]) <= row["high"]

    def test_normalize_chinese_columns(self):
        raw = pd.DataFrame({
            "日期": ["2026-09-01"],
            "开盘价": [10.0], "最高价": [11.0], "最低价": [9.5], "收盘价": [10.5],
            "成交量": [1234],
        })
        out = AkshareClient._normalize(raw, "RB.SHF")
        assert list(out.columns) == STD_COLUMNS
        assert out.iloc[0]["close"] == 10.5
        assert out.iloc[0]["volume"] == 1234.0

    def test_normalize_drops_degenerate_rows(self):
        raw = pd.DataFrame({
            "date": ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04"],
            "open": [10.0, 10.0, 10.0, 10.0],
            "high": [11.0, 11.0, 9.0, 11.0],     # 09-03: high<low 自相矛盾
            "low": [9.5, 9.5, 9.8, 9.5],
            "close": [10.5, -0.32, 10.0, 10.0],   # 09-02: 非正价格（前复权退化）
            "volume": [100, 100, 100, 100],
        })
        out = AkshareClient._normalize(raw, "X.SH")
        assert len(out) == 2
        assert set(out["date"]) == {"2026-09-01", "2026-09-04"}

    def test_normalize_volume_placeholder_for_fx(self):
        """外汇无成交量：如实填 1.0（不编造价格，只防下游除零）。"""
        raw = pd.DataFrame({
            "date": ["2026-09-09"],
            "open": [6.70], "low": [6.69], "high": [6.72], "close": [6.71],
        })
        out = AkshareClient._normalize(raw, "USDCNH.FXCM")
        assert out.iloc[0]["volume"] == 1.0

    def test_normalize_sorts_and_dedups(self):
        raw = pd.DataFrame({
            "date": ["2026-09-03", "2026-09-01", "2026-09-01"],
            "open": [10.0, 10.0, 10.0], "high": [11.0, 11.0, 11.0],
            "low": [9.0, 9.0, 9.0], "close": [10.5, 10.5, 10.5],
            "volume": [1, 1, 1],
        })
        out = AkshareClient._normalize(raw, "X.SH")
        assert out["date"].is_monotonic_increasing
        assert len(out) == 2

    def test_normalize_missing_required_columns_raises(self):
        raw = pd.DataFrame({"date": ["2026-09-01"], "open": [1.0]})
        with pytest.raises(ValueError):
            AkshareClient._normalize(raw, "X.SH")

    def test_clip_window(self):
        df = _raw(30)
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        out = AkshareClient._clip_window(df, "2024-01-05", "2024-01-10")
        assert out["date"].iloc[0] >= "2024-01-05"
        assert out["date"].iloc[-1] <= "2024-01-10"


# ----------------------------------------------------------------------
# 取数分派与降级（全部离线，注入 fake akshare）
# ----------------------------------------------------------------------
class _FakeAk:
    """假 akshare 模块：记录调用并返回预设结果/异常。"""

    def __init__(self, mapping=None):
        self.calls = []
        self.mapping = mapping or {}

    def _make(self, name):
        def _fn(**kwargs):
            self.calls.append((name, kwargs))
            value = self.mapping.get(name)
            if isinstance(value, Exception):
                raise value
            return value
        return _fn

    def __getattr__(self, name):
        return self._make(name)


class TestFetchDispatch:
    def test_fetch_stock_uses_sina_daily_qfq(self):
        c = AkshareClient({"data": {}})
        fake = _FakeAk({"stock_zh_a_daily": _raw(40)})
        c._ak = fake
        out = c.fetch("600519.SH", start_date="2024-01-01", end_date="2099-01-01")
        assert list(out.columns) == STD_COLUMNS
        name, kwargs = fake.calls[0]
        assert name == "stock_zh_a_daily"
        assert kwargs["symbol"] == "sh600519"
        assert kwargs["adjust"] == "qfq"
        assert kwargs["start_date"] == "20240101"

    def test_fetch_etf_uses_sina_fund_hist(self):
        c = AkshareClient({"data": {}})
        fake = _FakeAk({"fund_etf_hist_sina": _raw(40, cols=["date", "open", "high", "low", "close", "volume", "amount"])})
        c._ak = fake
        out = c.fetch("510300.SH")
        assert out is not None and len(out) == 40
        assert fake.calls[0][0] == "fund_etf_hist_sina"
        assert fake.calls[0][1]["symbol"] == "sh510300"

    def test_fetch_futures_uses_sina_daily_kline(self, monkeypatch):
        """期货走新浪直连通道（复用 Session + 浏览器头），修复 456 限流。"""
        c = AkshareClient({"data": {}})
        c._ak = _FakeAk({})
        calls = {}

        def _fake_futures(code):
            calls["code"] = code
            return pd.DataFrame({
                "date": ["2026-09-09", "2026-09-10"],
                "open": [3165.0, 3142.0],
                "high": [3167.0, 3143.0],
                "low": [3137.0, 3100.0],
                "close": [3146.0, 3108.0],
                "volume": [725825.0, 932720.0],
            })

        monkeypatch.setattr(c, "_sina_futures_fetch", _fake_futures)
        out = c.fetch("RB.SHF")
        assert calls["code"] == "RB0"
        assert list(out.columns) == STD_COLUMNS
        assert out.iloc[-1]["close"] == pytest.approx(3108.0)

    def test_fetch_futures_sina_code_mapping(self):
        """交易所后缀 → 新浪主连代码，逐类覆盖。"""
        assert to_futures_sina_code("RB.SHF") == "RB0"
        assert to_futures_sina_code("I.DCE") == "I0"
        assert to_futures_sina_code("SR.CZC") == "SR0"
        assert to_futures_sina_code("SC.INE") == "SC0"

    def test_sina_futures_jsonp_parsing(self, monkeypatch):
        """新浪期货 JSONP 响应解析：切片必须从 "[" 起，否则只解出第 1 条记录。"""
        body = (
            "/*<script>location.href='//sina.com';</script>*/\n"
            "var _v=([{\"d\":\"2026-09-09\",\"o\":\"3165.000\",\"h\":\"3167.000\","
            "\"l\":\"3137.000\",\"c\":\"3146.000\",\"v\":\"725825\","
            "\"p\":\"1596393\",\"s\":\"3147.000\"},"
            "{\"d\":\"2026-09-10\",\"o\":\"3142.000\",\"h\":\"3143.000\","
            "\"l\":\"3100.000\",\"c\":\"3108.000\",\"v\":\"932720\","
            "\"p\":\"1600739\",\"s\":\"3117.000\"}]);"
        )

        class _Resp:
            text = body
            def raise_for_status(self):
                pass

        c = AkshareClient({"data": {}})
        monkeypatch.setattr(c, "_sina_get", lambda url, params=None: _Resp())
        df = c._sina_futures_fetch("RB0")
        assert df is not None
        assert len(df) == 2, "必须解析出全部记录（切片起点错误会只解出第 1 条）"
        assert list(df.columns)[:6] == ["date", "open", "high", "low", "close", "volume"]

    def test_sina_futures_anti_bot_page_returns_none(self, monkeypatch):
        """返回反爬页（非 JSONP）时如实返回 None，不抛异常。"""
        class _Resp:
            text = "<!Doctype html><html><head></head></html>"
            def raise_for_status(self):
                pass

        c = AkshareClient({"data": {}})
        monkeypatch.setattr(c, "_sina_get", lambda url, params=None: _Resp())
        assert c._sina_futures_fetch("CU0") is None

    def test_retry_then_success(self, monkeypatch):
        """首次限流失败 → 退避重试后成功（新浪 456 场景）。"""
        c = AkshareClient({"data": {"akshare_max_retries": 2, "akshare_retry_backoff": 0.0}})
        c._ak = _FakeAk({})
        calls = {"n": 0}

        def _flaky(ak, symbol, market, start, end):
            calls["n"] += 1
            if calls["n"] < 2:
                raise IndexError("list index out of range")  # 反爬页触发的解析异常
            return _raw(20)

        monkeypatch.setattr(c, "_fetch_equity", _flaky)
        out = c.fetch("600519.SH")
        assert out is not None and len(out) == 20
        assert calls["n"] == 2

    def test_retry_exhausted_returns_none(self, monkeypatch):
        c = AkshareClient({"data": {"akshare_max_retries": 1, "akshare_retry_backoff": 0.0}})
        c._ak = _FakeAk({})
        calls = {"n": 0}

        def _always_fail(ak, symbol, market, start, end):
            calls["n"] += 1
            raise IndexError("boom")

        monkeypatch.setattr(c, "_fetch_equity", _always_fail)
        assert c.fetch("600519.SH") is None
        assert calls["n"] == 2  # 1 次原始 + 1 次重试

    def test_fetch_forex_uses_sina_fx_endpoint(self, monkeypatch):
        c = AkshareClient({"data": {}})
        fake = _FakeAk({})
        c._ak = fake

        raw = pd.DataFrame({
            "date": ["2026-09-09", "2026-09-10"],
            "open": [6.7066, 6.7064], "low": [6.7022, 6.7037],
            "high": [6.7077, 6.7150], "close": [6.7064, 6.7145],
        })
        calls = {}

        def _fake_sina(sym):
            calls["sym"] = sym
            return raw

        monkeypatch.setattr(c, "_sina_fx_fetch", _fake_sina)
        out = c.fetch("USDCNH.FXCM")
        assert calls["sym"] == "fx_susdcnh"
        assert len(out) == 2
        assert out.iloc[-1]["close"] == pytest.approx(6.7145)

    def test_fetch_unknown_symbol_returns_none(self):
        c = AkshareClient({"data": {}})
        c._ak = _FakeAk({})
        assert c.fetch("XXX.QQ") is None

    def test_fetch_unsupported_fx_returns_none(self):
        c = AkshareClient({"data": {}})
        c._ak = _FakeAk({})
        assert c.fetch("ZZZ.FXCM") is None

    def test_fetch_akshare_not_installed_returns_none(self):
        """akshare 不可导入时必须静默返回 None（不抛异常）。"""
        c = AkshareClient({"data": {}})
        c._get_ak = lambda: None
        assert c.fetch("600519.SH") is None
        assert c.fetch("RB.SHF") is None
        assert c.fetch("USDCNH.FXCM") is None

    def test_get_ak_lazy_import_failsoft(self, monkeypatch):
        """_get_ak 内部 ImportError 分支：未安装 akshare → None（不炸）。"""
        import builtins
        real_import = builtins.__import__

        def _boom(name, *a, **k):
            if name == "akshare":
                raise ImportError("not installed")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _boom)
        assert AkshareClient({"data": {}})._get_ak() is None

    def test_fetch_api_exception_swallowed(self):
        c = AkshareClient({"data": {}})
        c._ak = _FakeAk({"stock_zh_a_daily": RuntimeError("boom")})
        assert c.fetch("600519.SH") is None

    def test_fetch_empty_result_returns_none(self):
        c = AkshareClient({"data": {}})
        c._ak = _FakeAk({"stock_zh_a_daily": pd.DataFrame()})
        assert c.fetch("600519.SH") is None

    def test_fx_fallback_can_be_disabled(self):
        c = AkshareClient({"data": {"akshare_sina_fx_fallback": False}})
        c._ak = _FakeAk({})
        assert c.fetch("USDCNH.FXCM") is None

    def test_akshare_available_is_bool(self):
        assert isinstance(akshare_available(), bool)


# ----------------------------------------------------------------------
# DataCollector 回退链（akshare 位于 wind 之后、tencent 之前）
# ----------------------------------------------------------------------
class _StubCollector(DataCollector):
    """记录各源调用顺序的采集器；不触网、不落盘。"""

    def __init__(self, config, results):
        super().__init__(config)
        self.results = results
        self.order = []

    def _save_cache(self, symbol, df):  # noqa: D102
        pass

    def _fetch_wind(self, symbol):  # noqa: D102
        self.order.append("wind")
        return self.results.get("wind")

    def _fetch_akshare(self, symbol, start_date="", end_date=""):  # noqa: D102
        self.order.append("akshare")
        return self.results.get("akshare")

    def _fetch_tencent(self, symbol, start_date="", end_date=""):  # noqa: D102
        self.order.append("tencent")
        return self.results.get("tencent")


def _cfg(sources):
    return {
        "data": {"source": list(sources), "raw_dir": "/tmp/s14_stub", "quality_gate": False},
        "training": {},
    }


class TestFallbackChain:
    def test_chain_order_is_wind_akshare_tencent(self):
        c = _StubCollector(_cfg(["wind", "akshare", "tencent", "simulation"]), {})
        c._fetch_simulation = lambda symbol, n=300: None
        c._fetch_with_fallback("X.SH")
        assert c.order == ["wind", "akshare", "tencent"]

    def test_akshare_short_circuits_tencent(self):
        df = _raw(30)
        c = _StubCollector(_cfg(["wind", "akshare", "tencent", "simulation"]),
                           {"akshare": df})
        out = c._fetch_with_fallback("X.SH")
        assert out is not None and len(out) == 30
        # wind 失败 → akshare 成功 → 不再走 tencent
        assert c.order == ["wind", "akshare"]

    def test_falls_through_to_tencent_when_akshare_fails(self):
        df = _raw(30)
        c = _StubCollector(_cfg(["wind", "akshare", "tencent", "simulation"]),
                           {"akshare": None, "tencent": df})
        out = c._fetch_with_fallback("X.SH")
        assert out is not None
        assert c.order == ["wind", "akshare", "tencent"]

    def test_realtime_skips_simulation_and_uses_akshare(self):
        df = _raw(30)
        c = _StubCollector(_cfg(["wind", "akshare", "tencent", "simulation"]),
                           {"akshare": df})
        out = c.fetch_realtime("X.SH")
        assert out is not None
        assert c.order == ["wind", "akshare"]
        assert "simulation" not in c.order

    def test_akshare_client_import_error_is_failopen(self, monkeypatch):
        """未安装 akshare 时 _fetch_akshare 必须静默返回 None，不向上抛。"""
        import src.data.collector as collector_mod

        orig_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def _boom(name, *a, **k):
            if name.endswith("akshare_client"):
                raise ImportError("no akshare_client")
            return orig_import(name, *a, **k)

        monkeypatch.setattr("builtins.__import__", _boom)
        c = DataCollector(_cfg(["akshare"]))
        assert c._fetch_akshare("600519.SH") is None


# ----------------------------------------------------------------------
# 配置（S14 口径）
# ----------------------------------------------------------------------
class TestConfigWiring:
    def _load(self, name):
        import yaml
        with open(PROJECT_ROOT / "configs" / name, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_pro_source_chain_promotes_akshare(self):
        cfg = self._load("config_pro.yaml")
        assert cfg["data"]["source"] == ["wind", "akshare", "tencent", "simulation"]

    def test_pro_futures_and_forex_enabled(self):
        cfg = self._load("config_pro.yaml")
        assert cfg["data"]["markets"]["futures"]["enabled"] is True
        assert cfg["data"]["markets"]["forex"]["enabled"] is True
        assert cfg["data"]["markets"]["futures"]["symbols"]
        assert cfg["data"]["markets"]["forex"]["symbols"]


# ----------------------------------------------------------------------
# akshare 依赖声明（P1：从可选升为直接依赖）
# ----------------------------------------------------------------------
class TestRequirements:
    def test_akshare_declared_in_requirements(self):
        text = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
        assert "akshare" in text, "akshare 升为 P1 后必须写入 requirements.txt"

    def test_no_gate_side_effect(self):
        """数据源升级只影响取数，不得改动任何门禁配置。"""
        import yaml
        with open(PROJECT_ROOT / "configs" / "config_pro.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        gate = cfg.get("strategy_gate", {}) or {}
        # 门禁开关不由本次改动引入/翻转；若存在则必须显式布尔
        for key in ("enabled", "use_audit"):
            if key in gate:
                assert isinstance(gate[key], bool)
