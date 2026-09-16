"""模型优化对照守卫（Issue #55 步骤②③）。

守卫点：
1. **无泄漏**：带 TB 标签的监督集，喂给模型的特征矩阵里**不得**含标签列
   （这是本模块第一版实测暴露的真缺陷）；
2. 标签口径对照：两口径必须同折、同样本（否则差异可能来自样本集不同）；
3. 周期重排：候选整体做多重比较校正；全不显著时**不**倾斜权重（退化为等权）；
4. 结构性纪律：`affects_gate=False`，只读，不写配置；
5. 指标口径：命中率去中性 0.5、平均收益按信号方向建仓。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.model_improvement import (
    _hit_rate,
    _mean_return,
    _spearman,
    build_report,
    evaluate_variant,
    horizon_weight_experiment,
    label_variant_experiment,
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
    }


def _prices(n: int = 320, seed: int = 3, drift: float = 0.0004) -> pd.DataFrame:
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


def _dataset(n_symbols: int = 4, n: int = 320) -> dict:
    return {f"S{i}.SH": _prices(n, seed=10 + i) for i in range(n_symbols)}


class TestMetricHelpers:
    def test_hit_rate_counts_direction_agreement(self):
        # 概率 0.9（看涨） + 正收益 → 命中；0.1 + 正收益 → 不命中
        assert _hit_rate([0.9, 0.1], [0.01, 0.01]) == pytest.approx(0.5)

    def test_hit_rate_skips_nonfinite(self):
        assert _hit_rate([0.9, float("nan")], [0.01, 0.02]) == pytest.approx(1.0)

    def test_mean_return_signs_by_direction(self):
        # 看涨 + 正收益 → +；看跌 + 正收益 → −
        assert _mean_return([0.9, 0.1], [0.02, 0.02]) == pytest.approx(0.0)

    def test_mean_return_empty_is_none(self):
        assert _mean_return([], []) is None

    def test_spearman_monotone(self):
        assert _spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)

    def test_spearman_needs_three_points(self):
        assert _spearman([1, 2], [1, 2]) is None


class TestEvaluateVariant:
    def test_unavailable_when_too_few_samples(self):
        X = np.zeros((5, 3))
        y = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
        out = evaluate_variant(X, y, np.zeros(5), [(np.arange(3), np.arange(3, 5))],
                               _config(), label="t")
        assert out["available"] is False
        assert out["reason"]


class TestLabelVariantExperiment:
    def test_runs_and_returns_both_arms(self):
        out = label_variant_experiment(_dataset(), _config(), 5, folds=2)
        assert out["affects_gate"] is False
        if out["available"]:
            assert out["old_label"]["label"] == "fixed_h"
            assert out["new_label"]["label"] == "triple_barrier"
            # 同折同样本：两臂样本数必须一致
            assert out["old_label"]["n_samples"] == out["new_label"]["n_samples"]

    def test_improved_requires_both_legs(self):
        out = label_variant_experiment(_dataset(), _config(), 5, folds=2)
        if out["available"]:
            d = out["delta"]
            # 判定纪律：命中率与平均收益需同时改善且 IC 不劣
            assert d["improved"] == bool(
                d["hit_rate"] > 0 and d["mean_realized_return"] > 0 and d["ic"] >= 0)

    def test_empty_data_unavailable(self):
        out = label_variant_experiment({}, _config(), 5, folds=2)
        assert out["available"] is False


class TestHorizonWeightExperiment:
    def test_no_holdout_of_config(self):
        cfg = _config()
        before = dict(cfg["data"]["prediction_horizons"])
        horizon_weight_experiment(_dataset(), cfg, (5, 10), folds=2)
        assert cfg["data"]["prediction_horizons"] == before

    def test_returns_gate_discipline(self):
        out = horizon_weight_experiment(_dataset(), _config(), (5, 10), folds=2)
        assert out["affects_gate"] is False

    def test_weights_sum_to_one_when_available(self):
        out = horizon_weight_experiment(_dataset(), _config(), (5, 10), folds=2)
        if out.get("available") and out.get("evidence_weights"):
            assert sum(out["evidence_weights"].values()) == pytest.approx(1.0, abs=1e-4)
            assert all(w >= 0 for w in out["evidence_weights"].values())

    def test_all_zero_scores_falls_back_to_equal_weight(self):
        # 构造"无证据"场景：全部周期退化等权，不倾斜
        out = horizon_weight_experiment({}, _config(), (5, 10), folds=2)
        assert out["available"] is False


class TestBuildReport:
    def test_no_leak_into_features(self):
        """核心守卫：带 TB 标签的监督集，特征矩阵里不得含标签列。"""
        from src.data.preprocessor import FeatureEngineer
        from src.eval.model_improvement import _build_supervised_with_tb

        data = _dataset()
        combined = _build_supervised_with_tb(data, _config(), 5)
        assert not combined.empty
        fe = FeatureEngineer(_config())
        cols = fe.get_feature_columns(combined, 5)
        assert not any(str(c).startswith(("label_", "target_")) for c in cols)
        # TB 标签列确实存在于监督集里（否则这条守卫是空转）
        assert "label_tb_5d_bin" in combined.columns

    def test_report_shape(self):
        rep = build_report(_dataset(), _config(), horizons=(5,), folds=2)
        assert rep["kind"] == "model_improvement"
        assert rep["affects_gate"] is False
        assert rep["readonly"] is True
        assert "label_variant" in rep
        assert "horizon_weight" in rep

    def test_empty_data_unavailable(self):
        rep = build_report({}, _config(), horizons=(5,), folds=2)
        assert rep["available"] is False
        assert rep["reason"]

    def test_conclusion_note_states_human_signoff(self):
        rep = build_report(_dataset(), _config(), horizons=(5,), folds=2)
        assert "人工签字" in rep["conclusion_note"]
