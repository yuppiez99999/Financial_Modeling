"""S13（G3）qlib 因子库接入测试：数据层 / Alpha158 / provider / 增量 A/B。

测试要点（对照 00_kickoff/leakage_checklist.md 与 README §18A 统一原则）：
  - .bin 转换：行序升序、退化行剔除、目录结构符合 qlib 标准；
  - Alpha158：纯 pandas 实现、列数与 qlib 官方一致（8 形态 + 18×5 窗口 = 98）、
    **无前视**（篡改尾部价格不影响历史因子值）、warmup 期如实为 NaN；
  - QlibFactorProvider：短数据 → 中性 0 + available=False（不编造）、
    有界 (-1,1)、族映射与 FactorLibrary 同构、fail-soft；
  - A/B：两臂同折、只差特征列、affects_gate 恒为 False、样本不足如实 unavailable；
  - CLI：`qlib-ab` 命令注册在 choices 中。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from integrations.qlib.alpha158 import alpha158_columns, compute_alpha158
from integrations.qlib.data_layer import dump_all, to_qlib_dataframe
from src.factors.qlib_factor_provider import (
    A158_FAMILIES,
    PROVIDER_FACTOR_COLUMNS,
    QlibFactorProvider,
)
from src.eval.qlib_ab import QlibABExperiment, build_ab_report


def _make_market(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, n)))
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": close * (1 + rng.normal(0, 0.003, n)),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    })


# ----------------------------------------------------------------------
# T13.1 数据层：CSV → qlib .bin
# ----------------------------------------------------------------------
class TestDataLayer:
    def test_to_qlib_dataframe_sorts_and_dedups(self):
        df = _make_market(120)
        df_shuffled = df.sample(frac=1.0, random_state=1)
        std = to_qlib_dataframe(df_shuffled, "TEST.SH")
        assert std is not None
        assert std["date"].is_monotonic_increasing
        assert len(std) == 120
        assert "factor" in std.columns

    def test_to_qlib_dataframe_missing_columns(self):
        df = _make_market(100).drop(columns=["volume"])
        assert to_qlib_dataframe(df, "X") is None

    def test_to_qlib_dataframe_insufficient_rows(self):
        assert to_qlib_dataframe(_make_market(30), "X") is None

    def test_dump_all_directory_structure(self, tmp_path):
        data = {"AAA.SH": _make_market(120), "BBB.SH": _make_market(120)}
        out = dump_all(tmp_path / "bin", data)
        assert out["symbols_written"] == 2
        assert out["affects_gate"] is False
        feats = tmp_path / "bin" / "features" / "AAA.SH"
        for field in ("open", "high", "low", "close", "volume"):
            assert (feats / f"{field}.day.bin").exists()
        assert (tmp_path / "bin" / "calendars" / "day.txt").exists()
        assert (tmp_path / "bin" / "instruments" / "all.txt").exists()

    def test_dump_all_skips_invalid_symbol_failsoft(self, tmp_path):
        bad = _make_market(100).assign(volume=0.0)  # 退化成交量 → 全行剔除
        out = dump_all(tmp_path / "bin", {"BAD.SH": bad, "OK.SH": _make_market(100)})
        assert "OK.SH" in out["written"]
        assert "BAD.SH" in out["skipped"]

    def test_bin_values_roundtrip_order(self, tmp_path):
        df = _make_market(120)
        dump_all(tmp_path / "bin", {"X.SH": df})
        content = (tmp_path / "bin" / "features" / "X.SH" / "close.day.bin").read_text()
        lines = [ln for ln in content.strip().split("\n") if ln]
        assert lines[0].startswith("0\t")  # 日序号从 0 起
        first_val = float(lines[0].split("\t")[1])
        assert first_val == pytest.approx(df["close"].iloc[0], rel=1e-6)


# ----------------------------------------------------------------------
# T13.1/T13.2 Alpha158：列完整性 + 无前视
# ----------------------------------------------------------------------
class TestAlpha158:
    def test_column_count_matches_qlib(self):
        # qlib Alpha158（windows 5/10/20/30/60）：8 形态 + 18 项 × 5 窗口 = 98
        assert len(alpha158_columns()) == 98

    def test_no_lookahead_tail_tamper(self):
        df = _make_market(400)
        out = compute_alpha158(df)
        cols = list(out.columns)
        tampered = df.copy()
        tampered.loc[300:, "close"] = 99_999.0
        out2 = compute_alpha158(tampered)
        # 篡改 300 行之后 → 第 200 行因子值必须逐一相等
        assert np.allclose(
            out.loc[200, cols], out2.loc[200, cols], equal_nan=True, rtol=1e-12)

    def test_warmup_rows_are_nan(self):
        df = _make_market(200)
        out = compute_alpha158(df)
        # 前 59 行（60 日窗口未满）至少 60 日窗口的列应为 NaN
        assert out["A158_MA60"].iloc[:59].isna().all()
        assert out["A158_MA60"].iloc[59:].notna().all()

    def test_insufficient_history_returns_empty(self):
        out = compute_alpha158(_make_market(30))
        assert out.empty

    def test_slope_rsquare_resi_consistency(self):
        """完美线性序列：斜率=斜率真值、R²=1、残差≈0。"""
        n = 100
        df = pd.DataFrame({
            "open": np.arange(1, n + 1, dtype=float),
            "high": np.arange(1, n + 1, dtype=float),
            "low": np.arange(1, n + 1, dtype=float),
            "close": np.arange(1, n + 1, dtype=float),
            "volume": np.ones(n),
        })
        out = compute_alpha158(df)
        assert out["A158_BETA20"].iloc[-1] == pytest.approx(1.0, abs=1e-9)
        assert out["A158_RSQR20"].iloc[-1] == pytest.approx(1.0, abs=1e-9)
        assert out["A158_RESI20"].iloc[-1] == pytest.approx(0.0, abs=1e-9)


# ----------------------------------------------------------------------
# T13.2 QlibFactorProvider
# ----------------------------------------------------------------------
class TestQlibFactorProvider:
    def test_compute_appends_factor_columns(self):
        df = _make_market(400)
        provider = QlibFactorProvider({})
        out = provider.compute(df)
        for col in PROVIDER_FACTOR_COLUMNS:
            assert col in out.columns
        assert "factor_a158_available" in out.columns

    def test_bounded_range(self):
        out = QlibFactorProvider({}).compute(_make_market(400))
        for col in PROVIDER_FACTOR_COLUMNS:
            assert out[col].abs().max() <= 1.0 + 1e-9

    def test_short_data_returns_neutral(self):
        out = QlibFactorProvider({}).compute(_make_market(30))
        assert (out["factor_a158_available"] == 0).all()
        for col in PROVIDER_FACTOR_COLUMNS:
            assert (out[col] == 0).all()

    def test_no_lookahead(self):
        df = _make_market(400)
        out = QlibFactorProvider({}).compute(df)
        tampered = df.copy()
        tampered.loc[350:, "close"] = 12345.0
        out2 = QlibFactorProvider({}).compute(tampered)
        assert np.allclose(
            out.loc[200, list(PROVIDER_FACTOR_COLUMNS)],
            out2.loc[200, list(PROVIDER_FACTOR_COLUMNS)], equal_nan=True)

    def test_family_map_structure(self):
        fm = QlibFactorProvider.family_map(None)
        assert set(fm) == set(A158_FAMILIES)
        assert all(cols == [f"factor_a158_{k}"] for k, cols in fm.items())

    def test_factor_columns_static(self):
        df = _make_market(200)
        out = QlibFactorProvider({}).compute(df)
        cols = QlibFactorProvider.factor_columns(out)
        assert cols == [c for c in PROVIDER_FACTOR_COLUMNS if c in out.columns]


# ----------------------------------------------------------------------
# T13.3 增量 A/B
# ----------------------------------------------------------------------
def _make_supervised(n: int = 600, seed: int = 11) -> pd.DataFrame:
    """构造含特征 / target / _fwd_ret / factor_a158_* 的监督集（无前视）。"""
    rng = np.random.default_rng(seed)
    df = _make_market(n, seed=seed)
    out = QlibFactorProvider({}).compute(df)
    out["f1"] = rng.normal(0, 1, n)
    out["f2"] = rng.normal(0, 1, n)
    out["_fwd_ret"] = out["close"].shift(-10) / out["close"] - 1
    out["target_10d"] = (out["_fwd_ret"] > 0).astype(float)
    out.loc[out["_fwd_ret"].isna(), "target_10d"] = np.nan
    return out.dropna(subset=["target_10d"]).reset_index(drop=True)


LIGHTGBM_TEST_CONFIG = {
    "model": {"lightgbm": {
        "objective": "binary", "n_estimators": 30, "learning_rate": 0.1,
        "max_depth": 3, "num_leaves": 7, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 0.1, "verbose": -1,
    }},
    "training": {"save_dir": "models"},
}


class TestQlibAB:
    def _splits(self, n):
        cut = int(n * 0.7)
        return [(np.arange(0, cut), np.arange(cut, n))]

    def test_affects_gate_always_false(self):
        combined = _make_supervised()
        exp = QlibABExperiment(LIGHTGBM_TEST_CONFIG)
        res = exp.compare(
            combined, 10, ["f1", "f2"], list(PROVIDER_FACTOR_COLUMNS),
            self._splits(len(combined)))
        assert res["affects_gate"] is False

    def test_compare_runs_both_arms_same_folds(self):
        combined = _make_supervised()
        exp = QlibABExperiment(LIGHTGBM_TEST_CONFIG)
        res = exp.compare(
            combined, 10, ["f1", "f2"], list(PROVIDER_FACTOR_COLUMNS),
            self._splits(len(combined)))
        assert res["available"]
        assert res["base"]["samples"] == res["with_a158"]["samples"]
        assert res["base"]["folds"] == res["with_a158"]["folds"]
        assert res["a158_features"] == 2 + len(PROVIDER_FACTOR_COLUMNS)

    def test_missing_a158_columns_unavailable(self):
        combined = _make_supervised().drop(columns=["factor_a158_trend"])
        res = QlibABExperiment(LIGHTGBM_TEST_CONFIG).compare(
            combined, 10, ["f1", "f2"], list(PROVIDER_FACTOR_COLUMNS), [(np.arange(0, 10), np.arange(10, 20))])
        assert res["available"] is False
        assert res["reason"] == "missing_a158_columns"

    def test_missing_target_unavailable(self):
        combined = _make_supervised().drop(columns=["target_10d"])
        res = QlibABExperiment(LIGHTGBM_TEST_CONFIG).compare(
            combined, 10, ["f1", "f2"], list(PROVIDER_FACTOR_COLUMNS), [(np.arange(0, 10), np.arange(10, 20))])
        assert res["available"] is False

    def test_insufficient_samples_unavailable(self):
        combined = _make_supervised(80).head(20)
        res = QlibABExperiment(LIGHTGBM_TEST_CONFIG).compare(
            combined, 10, ["f1", "f2"], list(PROVIDER_FACTOR_COLUMNS),
            [(np.arange(0, 10), np.arange(10, len(combined)))])
        assert res["available"] is False
        assert res["reason"] == "insufficient_samples"

    def test_build_report_conservative_conclusion(self):
        res = {"10d": {"available": False, "reason": "x", "delta": None}}
        report = build_ab_report(res)
        assert report["affects_gate"] is False
        assert report["summary"]["any_improved"] is False
        assert "不纳入主线" in report["summary"]["conclusion"]


# ----------------------------------------------------------------------
# CLI 注册
# ----------------------------------------------------------------------
class TestCli:
    def test_qlib_ab_registered(self):
        import main as m

        parser = m.build_parser()
        args = parser.parse_args(["qlib-ab"])
        assert args.command == "qlib-ab"

    def test_run_qlib_ab_no_data_failsoft(self, tmp_path, monkeypatch):
        import main as m

        cfg = {
            "data": {"markets": {}, "prediction_horizons": {"short_term": {"days": 5}}},
            "strategy_gate": {"report_dir": str(tmp_path)},
        }
        monkeypatch.setattr(m, "_config_symbols", lambda c: ["NOPE.SH"])
        monkeypatch.setattr(
            "scripts.evaluate_models.load_market_data", lambda c, s, offline=False: {})
        out = m.run_qlib_ab(cfg, symbols=["NOPE.SH"])
        assert out.get("error")
        assert out.get("affects_gate") is False
