"""Q3 路线测试：实时数据流接入 + 信号一致性校验 + 评估口径修正。

覆盖点：
  - 快照解析 / 落盘 / 读取 / 清理（含 fail-open）
  - 滚动特征（严格无前视：截断尾部数据后前缀不变）
  - 盘中信号更新：baseline + live → intact / stale / unknown
  - 陈旧日K缓存拒绝给结论（防"拿过期数据装作实时"）
  - 一致性校验：跨周期 / 跨模型 / 跨口径
  - 评估口径：扣费后指标与保本胜率、盈亏比上限显式标记
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.streaming import (  # noqa: E402
    IntradaySnapshot,
    IntradayStore,
    MarketClock,
    RealtimeQuoteClient,
    RollingFeatureBuilder,
)
from src.inference.intraday import (  # noqa: E402
    MODEL_RULES_FALLBACK,
    IntradayPredictor,
)
from src.monitor.signal_consistency import (  # noqa: E402
    STATUS_CONSISTENT,
    STATUS_DIVERGENT,
    STATUS_UNKNOWN,
    SignalConsistencyChecker,
)


# ---------------------------------------------------------------- helpers
def _make_daily(bars: int = 80, start: str | None = None, end: str = "2026-04-10") -> pd.DataFrame:
    """构造确定性日K（正弦 + 线性趋势）。

    默认让最后一行落在"今天"（而非硬编码日期），否则随时间推移会触发
    `max_stale_days` 陈旧保护，使盘中用例变成时间炸弹。
    """
    start = start or (datetime.now() - timedelta(days=bars - 1)).strftime("%Y-%m-%d")
    dates = pd.date_range(start=start, periods=bars, freq="D")
    close = 100 + 5 * np.sin(np.arange(bars) / 5.0) + np.arange(bars) * 0.05
    return pd.DataFrame({
        "date": dates,
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.full(bars, 1_000_000),
    })


def _config(tmp_path: Path, **streaming) -> dict:
    base = {
        "streaming": {
            "enabled": True,
            "poll_seconds": 60,
            "realtime_dir": str(tmp_path / "realtime"),
            "retention_days": 30,
            "drift_notional": 0.01,
            "max_stale_days": 5,
            "min_bars": 30,
        },
        "consistency": {"tolerance": 0.15, "neutral_band": 0.02, "ignore_short_term": True},
        "data": {"raw_dir": str(tmp_path / "raw")},
    }
    base["streaming"].update(streaming)
    return base


# ---------------------------------------------------------------- streaming
def test_to_tencent_code_rejects_unsupported_market():
    from src.data.streaming import _to_tencent_code

    assert _to_tencent_code("600519.SH") == "sh600519"
    assert _to_tencent_code("000858.SZ") == "sz000858"
    # 期货 / 外汇腾讯不支持 → None（绝不编造快照）
    assert _to_tencent_code("RB.SHF") is None
    assert _to_tencent_code("USDCNH.FXCM") is None


def test_snapshot_change_pct_never_divides_by_zero():
    snap = IntradaySnapshot(symbol="X", ts="t", price=100.0, prev_close=0.0)
    assert snap.change_pct == 0.0  # 昨收缺失 → 0，不抛异常、不猜


def test_quote_client_parses_fields_and_skips_invalid():
    fields = ["1", "贵州茅台", "600519", "1700.00", "1650.00", "1660.00",
              "12345", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
              "", "", "", "", "", "", "", "", "1720.00", "1640.00"]
    snap = RealtimeQuoteClient._parse_fields("600519.SH", fields)
    assert snap is not None
    assert snap.price == 1700.0
    assert snap.prev_close == 1650.0
    assert round(snap.change_pct, 4) == round(50 / 1650, 4)

    # 价格非正 → None
    bad = list(fields)
    bad[3] = "0"
    assert RealtimeQuoteClient._parse_fields("600519.SH", bad) is None
    # 字段不足 → None
    assert RealtimeQuoteClient._parse_fields("600519.SH", ["1", "x"]) is None


def test_quote_client_failure_is_fail_open(monkeypatch):
    """实时源不可用 → 返回 {}，绝不抛异常（观测路径 fail-open）。"""
    class _Boom:
        def get(self, *a, **kw):
            raise RuntimeError("network down")

    client = RealtimeQuoteClient({})
    monkeypatch.setattr(client, "_get_session", lambda: _Boom())
    assert client.fetch_quotes(["600519.SH"]) == {}


def test_store_roundtrip_and_cleanup(tmp_path):
    store = IntradayStore(_config(tmp_path), base_dir=str(tmp_path / "realtime"))
    snaps = [
        IntradaySnapshot(symbol="600519.SH", ts="2026-04-10T10:00:00",
                         price=1700.0, prev_close=1650.0),
        IntradaySnapshot(symbol="000858.SZ", ts="2026-04-10T10:00:00",
                         price=200.0, prev_close=199.0),
    ]
    path = store.append(snaps)
    today = datetime.now().strftime("%Y-%m-%d")
    assert path is not None and path.exists()
    assert path.name == f"snapshots_{today}.jsonl"

    loaded = store.load()
    assert len(loaded) == 2
    assert [s.symbol for s in store.load("000858.SZ")] == ["000858.SZ"]
    assert store.load("UNKNOWN.SH") == []

    # 历史日期读取：显式落一份按日文件，验证按日过滤
    hist = store.path_for("2026-04-10")
    hist.write_text(json.dumps(snaps[0].to_dict(), ensure_ascii=False) + "\n", encoding="utf-8")
    assert len(store.load(day="2026-04-10")) == 1

    # 保留期清理
    old = store.path_for("2020-01-01")
    old.write_text('{"symbol":"X"}\n', encoding="utf-8")
    removed = store.cleanup()
    assert "snapshots_2020-01-01.jsonl" in removed
    assert not old.exists()


def test_store_survives_corrupt_line(tmp_path):
    store = IntradayStore(_config(tmp_path), base_dir=str(tmp_path / "realtime"))
    path = store.path_for("2026-04-10")
    path.write_text(
        '{"symbol":"600519.SH","ts":"t","price":1,"prev_close":1}\n'
        "not-json\n"
        '{"symbol":"000858.SZ","ts":"t","price":2,"prev_close":2}\n',
        encoding="utf-8",
    )
    assert len(store.load(day="2026-04-10")) == 2  # 单行损坏不影响其余


def test_snapshot_never_written_into_daily_cache(tmp_path):
    """实时快照目录必须与日K缓存目录分离（混入即污染训练集）。"""
    cfg = _config(tmp_path)
    store = IntradayStore(cfg)
    raw_dir = Path(cfg["data"]["raw_dir"])
    assert store.dir != raw_dir
    assert "realtime" in str(store.dir)


def test_market_clock_weekend_and_hours():
    sat = datetime(2026, 4, 11, 10, 0)  # 周六
    assert MarketClock.is_trading_hours(sat) is False
    wed_morning = datetime(2026, 4, 8, 10, 0)
    assert MarketClock.is_trading_hours(wed_morning) is True
    wed_lunch = datetime(2026, 4, 8, 12, 0)
    assert MarketClock.is_trading_hours(wed_lunch) is False


# ---------------------------------------------------------------- features
def test_rolling_features_no_lookahead():
    """截断尾部数据后，历史前缀特征不变（杜绝未来函数）。"""
    df = _make_daily(90)
    builder = RollingFeatureBuilder(_config(Path("/tmp")))
    full = builder.build("X", df)
    prefix = builder.build("X", df.iloc[:70])
    assert full.available and prefix.available
    # 70 行窗口的"最后一行"是 full 的第 70 行，其 ma_fast 必须与在 90 行上算出的对应值一致
    ma_at_70 = float(pd.Series(df["close"]).rolling(5).mean().iloc[69])
    assert prefix.ma_fast == pytest.approx(ma_at_70)


def test_rolling_features_insufficient_data_is_explicit():
    builder = RollingFeatureBuilder(_config(Path("/tmp")))
    feat = builder.build("X", _make_daily(10))
    assert feat.available is False
    assert "不足" in feat.reason


def test_rsi_bounds():
    df = _make_daily(80)
    feat = RollingFeatureBuilder(_config(Path("/tmp"))).build("X", df)
    assert 0.0 <= feat.rsi <= 100.0


# ---------------------------------------------------------------- intraday
def test_intraday_stale_when_intraday_move_contradicts_baseline(tmp_path):
    """baseline 看涨 + 盘中跌超阈值 → stale（观点可能失真）。"""
    cfg = _config(tmp_path)
    predictor = IntradayPredictor(cfg, engine=False)  # 强制规则口径
    df = _make_daily(90)

    class _Up:
        def predict_all_horizons(self, symbol):
            return {"predictions": {"long_term": {"direction": "看涨", "probability": 0.9}}}

    predictor._engine = _Up()
    snap = IntradaySnapshot(symbol="X", ts="now", price=95.0, prev_close=100.0)
    sig = predictor.build("X", df, snap)
    assert sig.available is True
    assert sig.drift["status"] == "stale"
    assert "看涨" in sig.drift["reason"]


def test_intraday_intact_when_move_agrees(tmp_path):
    cfg = _config(tmp_path)
    predictor = IntradayPredictor(cfg)
    predictor._engine = False
    df = _make_daily(90)

    class _Up:
        def predict_all_horizons(self, symbol):
            return {"predictions": {"long_term": {"direction": "看涨", "probability": 0.9}}}

    predictor._engine = _Up()
    snap = IntradaySnapshot(symbol="X", ts="now", price=102.0, prev_close=100.0)
    sig = predictor.build("X", df, snap)
    assert sig.drift["status"] == "intact"


def test_intraday_unknown_when_no_valid_move(tmp_path):
    """快照无有效涨跌（休市 / 昨收缺失）→ unknown，绝不误报。"""
    cfg = _config(tmp_path)
    predictor = IntradayPredictor(cfg)
    snapshot = IntradaySnapshot(symbol="X", ts="now", price=100.0, prev_close=0.0)

    class _Up:
        def predict_all_horizons(self, symbol):
            return {"predictions": {"long_term": {"direction": "看涨", "probability": 0.9}}}

    predictor._engine = _Up()
    sig = predictor.build("X", _make_daily(90), snapshot)
    assert sig.drift["status"] == "unknown"


def test_intraday_no_data_is_unavailable(tmp_path):
    predictor = IntradayPredictor(_config(tmp_path))
    sig = predictor.build("X", None, None)
    assert sig.available is False
    assert "无日K缓存" in sig.reason


def test_intraday_missing_snapshot_is_unavailable_but_explained(tmp_path):
    predictor = IntradayPredictor(_config(tmp_path))
    sig = predictor.build("X", _make_daily(90), None)
    assert sig.available is False
    assert "快照" in sig.reason


def test_intraday_stale_daily_cache_refuses_conclusion(tmp_path):
    """日K缓存落后超过 max_stale_days → 拒绝给盘中结论（不拿过期数据假装实时）。"""
    cfg = _config(tmp_path, max_stale_days=5)
    predictor = IntradayPredictor(cfg)
    old = _make_daily(90, start="2020-01-01")
    snap = IntradaySnapshot(symbol="X", ts="now", price=100.0, prev_close=99.0)
    sig = predictor.build("X", old, snap)
    assert sig.available is False
    assert "落后" in sig.reason


def test_engine_failure_degrades_to_rules_fallback(tmp_path):
    """模型加载失败 → 透明规则口径，并显式标注来源，不静默冒充模型输出。"""
    cfg = _config(tmp_path)

    class _Boom:
        def predict_all_horizons(self, symbol):
            raise RuntimeError("model missing")

    predictor = IntradayPredictor(cfg)
    predictor._engine = _Boom()
    sig = predictor.build("X", _make_daily(90), IntradaySnapshot("X", "now", 101.0, 100.0))
    assert sig.model == MODEL_RULES_FALLBACK
    assert sig.available is True


def test_poll_once_reports_stale_symbols(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    cfg["data"]["markets"] = {"stock": {"enabled": True, "symbols": ["600519.SH"]}}
    predictor = IntradayPredictor(cfg)

    class _Up:
        def predict_all_horizons(self, symbol):
            return {"predictions": {"long_term": {"direction": "看涨", "probability": 0.95}}}

    class _Client:
        def fetch_quotes(self, symbols):
            return {"600519.SH": IntradaySnapshot("600519.SH", "now", 90.0, 100.0)}

    class _Collector:
        def __init__(self, config):
            pass

        def load_cached(self, symbol):
            return _make_daily(90)

    predictor._engine = _Up()
    predictor.quote_client = _Client()
    monkeypatch.setattr("src.data.collector.DataCollector", _Collector)

    payload = predictor.poll_once(["600519.SH"])
    assert payload["quotes"] == 1
    assert payload["stale"] == ["600519.SH"]
    assert payload["signals"][0]["drift"]["status"] == "stale"


# ---------------------------------------------------------------- consistency
def test_horizon_consistency_detects_divergence():
    checker = SignalConsistencyChecker({"consistency": {"ignore_short_term": True}})
    pred = {"predictions": {
        "short_term": {"probability": 0.60},
        "mid_term": {"probability": 0.62},
        "long_term": {"probability": 0.38},
    }}
    detail = checker.check_horizons(pred)
    assert detail["status"] == STATUS_DIVERGENT
    assert "相悖" in detail["reason"]


def test_horizon_consistency_ignores_short_term_noise():
    """短周期噪声大：短期与中长周期相悖不判 divergent。"""
    checker = SignalConsistencyChecker({"consistency": {"ignore_short_term": True}})
    pred = {"predictions": {
        "short_term": {"probability": 0.40},
        "mid_term": {"probability": 0.60},
        "long_term": {"probability": 0.65},
    }}
    assert checker.check_horizons(pred)["status"] == STATUS_CONSISTENT


def test_horizon_consistency_only_short_term_falls_back():
    checker = SignalConsistencyChecker({"consistency": {"ignore_short_term": True}})
    pred = {"predictions": {"short_term": {"probability": 0.60}}}
    detail = checker.check_horizons(pred)
    assert detail["status"] == STATUS_CONSISTENT  # 仅有短期时以短期为准，不沉默


def test_horizon_consistency_neutral_band_is_unknown():
    checker = SignalConsistencyChecker({"consistency": {"neutral_band": 0.02}})
    pred = {"predictions": {"mid_term": {"probability": 0.501}, "long_term": {"probability": 0.499}}}
    assert checker.check_horizons(pred)["status"] == STATUS_UNKNOWN


def test_model_consistency_needs_two_components():
    checker = SignalConsistencyChecker()
    assert checker.check_models({"factor": 0.3})["status"] == STATUS_UNKNOWN
    detail = checker.check_models({"factor": 0.3, "tree": 0.25})
    assert detail["status"] == STATUS_CONSISTENT
    assert checker.check_models({"factor": 0.3, "tree": -0.2})["status"] == STATUS_DIVERGENT


def test_cross_caliber_detects_direction_conflict():
    checker = SignalConsistencyChecker({"consistency": {"tolerance": 0.15}})
    detail = checker.check_calibers(model_probability=0.62, factor_score=-0.3)
    assert detail["status"] == STATUS_DIVERGENT
    assert "相悖" in detail["reason"]


def test_cross_caliber_same_direction_but_gap_too_large():
    """同向但强度差距超容忍带也算分歧（"强烈看涨"配"微弱看涨"需复核）。"""
    checker = SignalConsistencyChecker({"consistency": {"tolerance": 0.10}})
    detail = checker.check_calibers(model_probability=0.70, factor_score=0.05)
    assert detail["status"] == STATUS_DIVERGENT
    assert "差距" in detail["reason"]


def test_cross_caliber_missing_data_is_explicit():
    checker = SignalConsistencyChecker()
    assert checker.check_calibers(None, 0.3)["status"] == STATUS_UNKNOWN
    assert checker.check_calibers(0.6, None)["status"] == STATUS_UNKNOWN


def test_consistency_report_aggregates_and_disclaims_gate():
    checker = SignalConsistencyChecker()
    report = checker.check(
        "600519.SH",
        predictions={"predictions": {"mid_term": {"probability": 0.65},
                                    "long_term": {"probability": 0.63}}},
        components={"tree": 0.4, "factor": -0.35},
        model_probability=0.64,
        factor_score=-0.35,
    ).to_dict()
    assert report["status"] == STATUS_DIVERGENT
    assert any("不参与策略门禁" in n for n in report["notes"])
    assert json.dumps(report, ensure_ascii=False)  # 可序列化


# ---------------------------------------------------------------- eval caliber
def test_evaluator_fee_aware_metrics_and_breakeven():
    from src.eval.evaluator import ModelEvaluator

    class _Model:
        def __init__(self, preds, probas):
            self._p = np.asarray(preds)
            self._pr = np.asarray(probas)

        def predict(self, X):
            return self._p

        def predict_proba(self, X):
            return self._pr

    y = np.array([1, 0, 1, 1, 0, 1])
    preds = np.array([1, 0, 1, 0, 0, 1])  # 5 对 1 错
    probas = np.array([[0.4, 0.6], [0.7, 0.3], [0.3, 0.7],
                       [0.6, 0.4], [0.7, 0.3], [0.2, 0.8]])
    model = _Model(preds, probas)

    res = ModelEvaluator({}).evaluate(model, np.zeros((6, 3)), y,
                                      horizon_name="short_term", horizon_days=5, fee=0.001)
    fin = res["metrics"]["financial"]
    # 等权 ±1 口径下必有亏损笔（1 错），故盈亏比是有限值而非哨兵
    assert fin["profit_factor_is_capped"] is False
    # 但扣费后净收益转负，扣费口径的盈亏比仍为有限值且更差
    assert fin["profit_factor_net"] <= fin["profit_factor"]
    assert fin["max_drawdown_unit"] == "trades"
    assert fin["sharpe_is_notional"] is True
    assert "profit_factor_net" in fin and "max_drawdown_net" in fin  # 扣费口径已产出
    assert fin["breakeven_win_rate"] == pytest.approx((1 + 0.002) / 2, abs=1e-6)
    # 毛口径胜率不受费用影响
    assert fin["win_rate"] == pytest.approx(5 / 6)


def test_evaluate_models_report_marks_profit_factor_cap(tmp_path):
    import scripts.evaluate_models as ev

    metrics = ev.compute_metrics(
        y_true=np.array([1, 1, 1]),
        proba=np.array([0.9, 0.8, 0.7]),
        fwd_ret=np.array([0.01, 0.02, 0.03]),
        horizon_days=5,
        fee=0.0005,
    )
    assert metrics["profit_factor_is_capped"] is True
    assert metrics["max_drawdown_unit"] == "cum_return"
    assert metrics["breakeven_win_rate"] == pytest.approx(0.5005)

    md = ev.render_markdown([{
        "status": "ok", "horizon": "short_term", "horizon_days": 5,
        "total_samples": 3, "feature_count": 10,
        "weighted": metrics,
        "folds": [],
    }], {"generated_at": "t", "model_type": "lightgbm", "symbol_count": 1,
         "folds": 1, "fee": 0.0005, "data_source": "test"})
    assert "已达解析上限" in md
    assert "保本胜率" in md


# ---------------------------------------------------------------- data hygiene
def test_tencent_parser_drops_non_positive_and_inverted_rows():
    """真实缺陷回归（2026-09-10）：腾讯前复权分页会返回非正价格。

    中国神华 2013 年段曾返回 912 行 close 为负 —— 会让 pct_change 产生 -inf，
    进而使 LightGBM 抛 `Input X contains infinity`，**整只标的预测全挂**。
    """
    rows = [
        ["2013-03-22", "-0.38", "-0.32", "-0.25", "-0.41", "104676"],   # 负价 → 剔除
        ["2013-03-25", "0.00", "0.03", "0.03", "-0.29", "205654"],      # close>0 但 low<0 → 剔除
        ["2013-03-27", "10.20", "10.10", "10.60", "10.40", "92948"],    # high<low → 剔除
        ["2013-03-26", "10.00", "10.50", "9.80", "10.20", "127283"],    # 正常 → 保留
    ]
    from src.data.tencent_client import TencentClient

    parsed = TencentClient._parse_rows(rows)
    assert len(parsed) == 1
    assert parsed["date"].iloc[0] == "2013-03-27"
    assert (parsed["close"] > 0).all()
    assert (parsed["high"] >= parsed["low"]).all()


def test_feature_pipeline_never_emits_infinity():
    """特征矩阵必须全有限：inf 会让 LightGBM 直接拒绝预测。"""
    from src.data.preprocessor import FeatureEngineer

    bars = 80
    dates = pd.date_range(end=pd.Timestamp(datetime.now().date()), periods=bars, freq="D")
    close = np.linspace(10, 20, bars)
    # 注入 0 价格（前复权退化值）→ pct_change 会产生 ±inf
    close[40] = 0.0
    df = pd.DataFrame({
        "date": dates, "open": close, "high": close * 1.01,
        "low": close * 0.99, "close": close, "volume": np.full(bars, 1e6),
    })
    fe = FeatureEngineer({"features": {"extended_indicators": True}})
    feat = fe.transform(df, horizon_days=5)
    for col in fe.get_feature_columns(feat, 5):
        values = np.asarray(feat[col], dtype=float)
        assert np.isfinite(values).all(), f"特征 {col} 含非有限值"


def test_nvi_guards_against_overflow():
    """NVI 单步乘法遇到极端收益率不得溢出为 inf。"""
    from src.data.indicators import TechnicalIndicators

    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    close = np.array([10.0] * 9 + [1e300])
    df = pd.DataFrame({
        "date": dates, "open": close, "high": close, "low": close,
        "close": close, "volume": np.array([1000.0] * 9 + [1.0]),
    })
    out = TechnicalIndicators.compute_all(df)
    assert np.isfinite(out["nvi"]).all()


# ---------------------------------------------------------------- API
def test_api_stream_status_and_signal_fail_open():
    """实时流未启用时接口返回 available=false 并给出原因，不抛 500、不估算。"""
    pytest.importorskip("fastapi")
    testclient = pytest.importorskip("starlette.testclient")
    from src.api import server

    app = server.create_app("configs/config.yaml")
    with testclient.TestClient(app) as client:
        r = client.get("/api/v1/stream/status")
        assert r.status_code == 200
        assert r.json()["available"] is False

        r = client.get("/api/v1/stream/600519.SH")
        assert r.status_code == 200
        body = r.json()
        assert body["symbol"] == "600519.SH"
        assert body["available"] is False
