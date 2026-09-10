"""Q1 排期：宏观指标数据源（MacroClient）测试。

覆盖：多源回退、本地缓存读写、asof 无前视对齐、显式零值降级、特征稳定性。
全部离线运行（不触网）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.macro_client import MacroClient, INDICATOR_REGISTRY
from src.data.preprocessor import FeatureEngineer


def _cfg(tmp_path, macro_enabled=False, sources=("local",)):
    with open(PROJECT_ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["macro"] = {"source": list(sources), "dir": str(tmp_path), "indicators": ["cpi", "pmi"]}
    cfg["features"]["macro_enabled"] = macro_enabled
    return cfg


# ---------- 注册表 ----------

def test_registry_covers_cpi_pmi_gdp():
    for key in ("cpi", "pmi", "gdp"):
        assert key in INDICATOR_REGISTRY
        assert INDICATOR_REGISTRY[key]["name"]


# ---------- 无数据：显式零值，不抛异常 ----------

def test_no_data_returns_empty_series(tmp_path):
    client = MacroClient(_cfg(tmp_path))
    series = client.get_series("cpi")
    assert isinstance(series, pd.Series)
    assert len(series) == 0


def test_features_zero_filled_when_unavailable(tmp_path):
    client = MacroClient(_cfg(tmp_path))
    feats = client.get_features()
    assert feats["macro_cpi_latest"] == 0.0
    assert feats["macro_pmi_yoy3"] == 0.0
    assert len(feats) == 2 * 3  # 2 指标 × (latest/mom/yoy3)


def test_health_reports_unavailable(tmp_path):
    health = MacroClient(_cfg(tmp_path)).health()
    assert health["cpi"]["status"] == "unavailable"
    assert health["cpi"]["points"] == 0


# ---------- 本地缓存 ----------

def test_seed_and_read_local_cache(tmp_path):
    client = MacroClient(_cfg(tmp_path))
    client.seed_from_dict("cpi", {
        "2024-01-01": 0.8, "2024-02-01": 0.7, "2024-03-01": 0.1,
    })
    assert (tmp_path / "macro_cpi.csv").exists()

    fresh = MacroClient(_cfg(tmp_path))  # 新实例从磁盘读
    series = fresh.get_series("cpi")
    assert len(series) == 3
    assert series.iloc[-1] == pytest.approx(0.1)


def test_get_features_computes_mom_and_yoy3(tmp_path):
    client = MacroClient(_cfg(tmp_path))
    client.seed_from_dict("cpi", {"2024-01-01": 1.0, "2024-02-01": 2.0, "2024-03-01": 3.0})
    feats = client.get_features(asof="2024-03-15")
    assert feats["macro_cpi_latest"] == pytest.approx(3.0)
    assert feats["macro_cpi_mom"] == pytest.approx(1.0)
    assert feats["macro_cpi_yoy3"] == pytest.approx(2.0)


def test_get_features_respects_asof(tmp_path):
    client = MacroClient(_cfg(tmp_path))
    client.seed_from_dict("cpi", {"2024-01-01": 1.0, "2024-06-01": 9.0})
    feats = client.get_features(asof="2024-03-01")
    assert feats["macro_cpi_latest"] == pytest.approx(1.0)  # 不使用 6 月数据


# ---------- attach_features 无前视 ----------

def test_attach_features_no_lookahead(tmp_path):
    client = MacroClient(_cfg(tmp_path))
    client.seed_from_dict("cpi", {"2024-01-01": 1.0, "2024-06-01": 5.0})

    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-15", "2024-03-15", "2024-07-15"]),
        "close": [1.0, 2.0, 3.0],
    })
    out = client.attach_features(df)
    assert list(out["macro_cpi_latest"]) == [1.0, 1.0, 5.0]  # 逐行 asof，无未来数据
    assert "macro_pmi_latest" in out.columns  # 无数据指标补零，列集合稳定
    assert out["macro_pmi_latest"].sum() == 0.0


def test_attach_features_missing_date_col_is_noop(tmp_path):
    client = MacroClient(_cfg(tmp_path))
    df = pd.DataFrame({"close": [1.0, 2.0]})
    out = client.attach_features(df)
    assert list(out.columns) == ["close"]


def test_attach_features_zero_when_all_unavailable(tmp_path):
    client = MacroClient(_cfg(tmp_path))
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3), "close": [1.0, 2.0, 3.0]})
    out = client.attach_features(df)
    assert (out["macro_cpi_latest"] == 0.0).all()


# ---------- 数据源回退 ----------

def test_unknown_source_is_skipped(tmp_path):
    client = MacroClient(_cfg(tmp_path, sources=("not-a-source", "local")))
    client.seed_from_dict("cpi", {"2024-01-01": 1.0})
    assert len(client.get_series("cpi")) == 1


def test_akshare_missing_dependency_degrades(tmp_path, monkeypatch):
    client = MacroClient(_cfg(tmp_path, sources=("akshare",)))
    # akshare 未安装 → 返回空序列而非抛异常
    assert len(client.get_series("cpi")) == 0


def test_akshare_normalize_columns():
    raw = pd.DataFrame({"月份": ["2024-01", "2024-02"], "今值": ["1.5", "2.5"]})
    series = MacroClient._normalize_akshare(raw)
    assert series is not None and len(series) == 2
    assert series.iloc[-1] == pytest.approx(2.5)
    assert MacroClient._normalize_akshare(pd.DataFrame({"a": [1]})) is None


def test_corrupt_cache_is_ignored(tmp_path):
    (tmp_path / "macro_cpi.csv").write_text("garbage\n1,2,3\n", encoding="utf-8")
    client = MacroClient(_cfg(tmp_path))
    assert len(client.get_series("cpi")) == 0


# ---------- 与特征工程集成 ----------

def _ohlcv(n=80):
    close = np.linspace(100, 130, n)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n),
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": np.full(n, 1000.0),
    })


def test_feature_engineer_injects_macro_when_enabled(tmp_path):
    cfg = _cfg(tmp_path, macro_enabled=True)
    MacroClient(cfg).seed_from_dict("cpi", {"2023-01-01": 0.5, "2024-01-01": 0.3})
    fe = FeatureEngineer(cfg)
    out = fe.transform(_ohlcv(), 5)
    cols = fe.get_feature_columns(out, 5)
    assert "macro_cpi_latest" in cols
    assert out["macro_cpi_latest"].iloc[-1] == pytest.approx(0.3)


def test_feature_engineer_skips_macro_when_disabled(tmp_path):
    cfg = _cfg(tmp_path, macro_enabled=False)
    fe = FeatureEngineer(cfg)
    out = fe.transform(_ohlcv(), 5)
    assert not [c for c in out.columns if c.startswith("macro_")]


def test_feature_engineer_survives_macro_failure(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, macro_enabled=True)
    fe = FeatureEngineer(cfg)

    class Boom:
        def attach_features(self, df, date_col="date"):
            raise RuntimeError("boom")

    monkeypatch.setattr(fe, "_get_macro_client", lambda: Boom())
    out = fe.transform(_ohlcv(), 5)  # 不抛异常
    assert "close" in out.columns
