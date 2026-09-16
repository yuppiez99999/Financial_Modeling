"""波动分层置信度有效性守卫（Issue #55 —— 修正"高置信=反向"的口径混淆）。

守卫点：
1. **只读纪律**：`affects_gate=False`，不改配置 / 门禁 / 权重；
2. **确定性**：同池打乱输入标的顺序，读数**必须**逐字段一致
   （否则 LightGBM 折内行序变化会让报告不可复算 —— 本模块实测暴露过）；
3. **分位数分层**：波动档用分位数切（不挑绝对阈值），三档全部给出、不隐藏；
4. **不吞样本**：任一分层样本不足 → 该档 `available=False` + reason，不外推；
5. **不改门槛**：置信度门槛是入参，本模块不选阈值、不改阈值；
6. **稳健性不虚报**：子池稳健性检验只报成功率与分级，不设"通过阈值"；
   不可用子池**不计入分母**（否则成功率会被可用子集稀释成假象）；
7. 指标口径与 `model_improvement` 同源（命中率去中性 0.5、平均收益按信号方向建仓）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.regime_conditioned_signal import (
    _build_supervised,
    _hit_rate,
    _ic,
    _mean_return,
    _spearman,
    build_report,
    confidence_diagnostics,
    regime_conditioned_evaluation,
    regime_stability_check,
    weight_reallocation_check,
)


def _config() -> dict:
    return {
        "model": {"lightgbm": {
            "objective": "binary", "n_estimators": 20, "learning_rate": 0.1,
            "max_depth": 3, "num_leaves": 7, "subsample": 0.9,
            "colsample_bytree": 0.9, "reg_alpha": 0.0, "reg_lambda": 0.0,
            "verbose": -1, "early_stopping_rounds": 5,
        }},
        "training": {"save_dir": "/tmp/trendcast_test_models"},
        "features": {"technical": {"ma_windows": [5, 10]}},
        "labeling": {},
        "data": {"prediction_horizons": {"short_term": 5, "mid_term": 10, "long_term": 20}},
        "signal": {"horizon_weights": {"short_term": 0.30, "mid_term": 0.35,
                                       "long_term": 0.35}},
    }


def _prices(n: int = 340, seed: int = 3, drift: float = 0.0004) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.015, n)
    close = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=n, freq="B"),
        "open": close * (1 + rng.normal(0, 0.002, n)),
        "high": close * (1 + np.abs(rng.normal(0, 0.005, n))),
        "low": close * (1 - np.abs(rng.normal(0, 0.005, n))),
        "close": close,
        "volume": rng.integers(1_000, 100_000, n).astype(float),
    })


def _dataset(n_symbols: int = 6, n: int = 340) -> dict:
    return {f"S{i}.SH": _prices(n, seed=10 + i) for i in range(n_symbols)}


class TestMetricHelpers:
    def test_hit_rate_direction_agreement(self):
        assert _hit_rate([0.9, 0.1], [0.01, 0.01]) == pytest.approx(0.5)

    def test_hit_rate_skips_nonfinite(self):
        assert _hit_rate([0.9, float("nan")], [0.01, 0.02]) == pytest.approx(1.0)

    def test_mean_return_signs_by_direction(self):
        assert _mean_return([0.9, 0.1], [0.02, 0.02]) == pytest.approx(0.0)

    def test_mean_return_empty_is_none(self):
        assert _mean_return([], []) is None

    def test_spearman_monotone(self):
        assert _spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)

    def test_spearman_needs_three_points(self):
        assert _spearman([1, 2], [1, 2]) is None

    def test_ic_ignores_nonfinite_pairs(self):
        # 5 个点里有 2 个非有限 → 剩 3 个有效点，仍可算
        assert _ic([0.1, 0.2, 0.3, 0.4, 0.5],
                   [float("nan"), 2.0, 3.0, float("inf"), 5.0]) is not None

    def test_ic_needs_three_valid_points(self):
        assert _ic([0.1, 0.2, 0.3], [float("nan"), 2.0, 3.0]) is None


class TestSupervisedBuild:
    def test_features_exclude_internal_and_label_columns(self):
        from src.eval.regime_conditioned_signal import _feature_columns
        ds = _build_supervised(_dataset(2), _config(), 5)
        cols = _feature_columns(ds, _config(), 5)
        assert not any(c.startswith("_") for c in cols)
        assert not any(c.startswith("target_") for c in cols)
        assert not any(c.startswith("label_") for c in cols)

    def test_deterministic_under_symbol_shuffle(self):
        """同池打乱标的顺序，监督集行序必须完全一致（复现性要求）。"""
        data = _dataset(6)
        ds1 = _build_supervised(data, _config(), 5)
        shuffled = {k: data[k] for k in reversed(list(data.keys()))}
        ds2 = _build_supervised(shuffled, _config(), 5)
        pd.testing.assert_frame_equal(
            ds1.reset_index(drop=True), ds2.reset_index(drop=True))

    def test_volatility_is_causal(self):
        """波动率只用截至 t 的信息：末行波动率在价格尾段改变前应不变。"""
        data = _dataset(2)
        ds1 = _build_supervised(data, _config(), 5)
        base = ds1["_vol"].iloc[100]
        # 改动 t=200 之后的收益，不应影响 t=100 处的 `_vol`
        d2 = {k: v.copy() for k, v in data.items()}
        for k in d2:
            d2[k].loc[200:, "close"] = d2[k].loc[200:, "close"] * 1.5
        ds2 = _build_supervised(d2, _config(), 5)
        assert ds2["_vol"].iloc[100] == pytest.approx(base, rel=1e-9)

    def test_empty_data(self):
        assert _build_supervised({}, _config(), 5).empty


class TestConfidenceDiagnostics:
    def test_too_few_samples_unavailable(self):
        ds = _build_supervised(_dataset(1, n=60), _config(), 5)
        proba = np.full(len(ds), np.nan)
        proba[:5] = 0.6
        out = confidence_diagnostics(ds, proba, 5)
        assert out["available"] is False
        assert out["reason"]

    def test_edge_signal_flag_requires_ret_ge_vol(self):
        """边际优势语义只在 IC(conf,收益) 为正且不劣于 IC(conf,波动) 时成立。"""
        ds = _build_supervised(_dataset(3), _config(), 5)
        rng = np.random.default_rng(0)
        proba = rng.uniform(0.2, 0.8, len(ds))
        out = confidence_diagnostics(ds, proba, 5)
        if out["available"]:
            ic_ret = out["ic_confidence_vs_realized_return"]
            ic_vol = out["ic_confidence_vs_forward_volatility"]
            assert out["is_edge_signal"] == bool(
                ic_ret is not None and ic_vol is not None
                and ic_ret > 0 and ic_ret >= ic_vol)


class TestRegimeConditionedEvaluation:
    def _proba(self, ds):
        rng = np.random.default_rng(1)
        return rng.uniform(0.25, 0.75, len(ds))

    def test_bands_are_quantile_based_and_all_reported(self):
        ds = _build_supervised(_dataset(8), _config(), 5)
        out = regime_conditioned_evaluation(ds, self._proba(ds), 5,
                                            confidence_thr=0.2)
        assert out["available"] is True
        names = [b["regime"] for b in out["bands"]]
        assert names == ["low_volatility", "mid_volatility", "high_volatility"]
        # 分位数切：低/高两档样本量应接近（不挑绝对阈值 → 不会退化成 1 个样本）
        low = next(b for b in out["bands"] if b["regime"] == "low_volatility")
        high = next(b for b in out["bands"] if b["regime"] == "high_volatility")
        assert abs(low["n_samples"] - high["n_samples"]) <= 2

    def test_threshold_is_input_not_chosen(self):
        ds = _build_supervised(_dataset(8), _config(), 5)
        out = regime_conditioned_evaluation(ds, self._proba(ds), 5,
                                            confidence_thr=0.42)
        assert out["confidence_threshold"] == pytest.approx(0.42)

    def test_high_threshold_unavailable_when_too_few(self):
        ds = _build_supervised(_dataset(6), _config(), 5)
        rng = np.random.default_rng(2)
        # 概率几乎贴 0.5 → 高置信子集为空
        proba = rng.uniform(0.49, 0.51, len(ds))
        out = regime_conditioned_evaluation(ds, proba, 5, confidence_thr=0.6)
        assert out["available"] is False
        assert out["reason"]

    def test_dirty_band_reports_reason_not_numbers(self):
        ds = _build_supervised(_dataset(6), _config(), 5)
        out = regime_conditioned_evaluation(ds, self._proba(ds), 5,
                                            confidence_thr=0.0)
        for band in out["bands"]:
            if not band["available"]:
                assert band["reason"]
                assert band["ic"] is None


class TestWeightReallocationCheck:
    def test_readonly_and_candidates_present(self):
        out = weight_reallocation_check(_dataset(6), _config(), (5, 10), folds=2)
        assert out["affects_gate"] is False
        if out["available"]:
            assert "current" in out["candidates"]
            assert "equal" in out["candidates"]

    def test_does_not_mutate_config_weights(self):
        cfg = _config()
        before = dict(cfg["signal"]["horizon_weights"])
        weight_reallocation_check(_dataset(6), cfg, (5, 10), folds=2)
        assert cfg["signal"]["horizon_weights"] == before


class TestStabilityCheck:
    def test_reports_rate_and_verdict(self):
        out = regime_stability_check(_dataset(8), _config(), horizon_days=5,
                                     folds=2, n_subsets=3, subset_size=4, seed=1,
                                     confidence_thr=0.0)
        # 合成分位数据下不一定每池都跑得出结论；有可用子池就必须给出率与分级
        if out["available"]:
            assert out["n_subsets_usable"] <= 3
            assert 0.0 <= out["hold_rate"] <= 1.0
            assert out["robustness_verdict"] in ("robust", "suggestive", "fragile")
        else:
            assert out["reason"]

    def test_unavailable_subsets_not_counted_in_denominator(self):
        out = regime_stability_check(_dataset(4), _config(), horizon_days=5,
                                     folds=2, n_subsets=2, subset_size=4, seed=0)
        # 分母只含真正跑出结论的子集
        assert out["n_holds"] <= out["n_subsets_usable"]

    def test_too_few_symbols_unavailable(self):
        out = regime_stability_check(_dataset(3), _config(), horizon_days=5,
                                     n_subsets=2, subset_size=18)
        assert out["available"] is False
        assert out["reason"]


class TestBuildReport:
    def test_readonly_flags_and_shape(self):
        report = build_report(_dataset(6), _config(), horizons=(5, 10), folds=2,
                              stability_subsets=0)
        assert report["affects_gate"] is False
        assert report["readonly"] is True
        assert set(report["confidence_diagnostics"].keys()) == {"5", "10"}
        assert set(report["regime_conditioned"].keys()) == {"5", "10"}

    def test_empty_data_unavailable(self):
        report = build_report({}, _config(), horizons=(5,), folds=2)
        assert report["available"] is False
        assert report["reason"]

    def test_does_not_change_gate_or_weights(self):
        cfg = _config()
        gate_before = dict(cfg.get("strategy_gate", {}) or {})
        weights_before = dict(cfg["signal"]["horizon_weights"])
        build_report(_dataset(6), cfg, horizons=(5,), folds=2, stability_subsets=0)
        assert (cfg.get("strategy_gate", {}) or {}) == gate_before
        assert cfg["signal"]["horizon_weights"] == weights_before

    def test_stability_horizon_is_middle(self):
        report = build_report(_dataset(6), _config(), horizons=(5, 10, 20),
                              folds=2, stability_subsets=1, stability_size=4)
        if "regime_stability" in report:
            assert report["regime_stability"]["horizon_days"] == 10
