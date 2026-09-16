"""特征集 × 模型族联合消融守卫（Issue #55 —— 最后一条未量化的嫌疑）。

守卫点：
1. **只读纪律**：`affects_gate=False`、`readonly=True`，不改特征集 / 模型配置 / 权重 / 门禁；
2. **唯一记分板**：判定只看「相对全池等权的净超额」，IC / 命中率 / AUC **不参与**判定；
   增量需 **净超额 > 0 且相对基线臂配对 t ≥ 2**；
3. **不夸大证据**：候选差异不显著 → `inconclusive`（不是 `no_increment`）；
   只有「净超额 ≤ 0 且 |t| ≥ 2」才允许说 `worse_than_baseline`；
4. **同批样本同折**：所有臂共享面板预生成的折索引（不是每臂各自切一次折）；
5. **分组不漏列**：未被任何族覆盖的特征列必须显式披露（fail-loud），
   且分组不能用 `in` 匹配（`ma_` 会吞掉 `ema_`/`dema_`/`wma_`/`trima_`）；
6. **模型族口径**：项目自研 `LightGBMModel` 入口是 `train()` 不是 `fit()` ——
   适配层必须把它收敛到与 sklearn 族同一口径，否则全臂静默"无样本"；
7. **不吞样本**：样本 / 调仓期数不足 → `available=False` + reason，不外推。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.feature_model_ablation import (
    FEATURE_GROUP_RULES,
    MIN_PERIODS,
    SUBSET_RECIPES,
    _arm_verdict,
    _subset_columns,
    _verdict_from,
    build_panel,
    build_report,
    group_feature_columns,
    make_model,
)


def _config() -> dict:
    return {
        "data": {"prediction_horizons": {"short_term": 5, "mid_term": 10,
                                         "long_term": 20}},
        "model": {"lightgbm": {"objective": "binary", "n_estimators": 20,
                               "learning_rate": 0.1, "max_depth": 3,
                               "num_leaves": 8, "subsample": 0.8,
                               "colsample_bytree": 0.8, "reg_alpha": 0.0,
                               "reg_lambda": 0.0, "verbose": -1,
                               "early_stopping_rounds": 5}},
        "training": {"save_dir": "reports/_ablation_models"},
        "features": {"extended_indicators": False, "macro_enabled": False,
                     "sentiment_enabled": False},
    }


def _prices(n: int = 320, seed: int = 3, drift: float = 0.0005) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.015, n)))
    return pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=n, freq="B"),
        "open": close, "high": close, "low": close, "close": close,
        "volume": 1e6,
    })


def _dataset(n_symbols: int = 6, n: int = 320) -> dict:
    return {f"S{i}.SH": _prices(n, seed=10 + i) for i in range(n_symbols)}


# ----------------------------------------------------------------------
# ① 特征分组
# ----------------------------------------------------------------------
class TestFeatureGrouping:
    def test_ma_prefix_does_not_swallow_ema_family(self):
        """`ma_` 用 `in` 匹配会吞掉 `ema_10`/`dema_5`/`wma_10`/`trima_5`（实测坑）。"""
        groups, uncovered = group_feature_columns(
            ["ma_5", "ma_5_ratio", "ema_10", "dema_5", "wma_10", "trima_5", "kama"])
        assert set(groups["ma"]) == {"ma_5", "ma_5_ratio"}
        assert set(groups["trend_extra"]) == {"ema_10", "dema_5", "wma_10", "trima_5"}
        assert groups["kama"] == ["kama"]
        assert uncovered == []

    def test_groups_are_mutually_exclusive_and_complete(self):
        feats = ["ret_1", "ma_5", "rsi", "macd", "boll_pos", "volatility",
                 "volume_ratio", "factor_value", "macro_cpi_latest", "kama",
                 "adx", "obv", "pivot", "doji"]
        groups, uncovered = group_feature_columns(feats)
        flat = [c for cols in groups.values() for c in cols]
        assert sorted(flat) == sorted(feats)
        assert uncovered == []

    def test_unknown_columns_are_disclosed_not_swallowed(self):
        groups, uncovered = group_feature_columns(["ret_1", "brand_new_signal"])
        assert uncovered == ["brand_new_signal"]
        # 未覆盖列不进任何子集（因此必须在报告里显式披露）
        panel = {"groups": groups, "features": ["ret_1", "brand_new_signal"]}
        assert "brand_new_signal" not in _subset_columns(panel, "returns_only")

    def test_every_recipe_resolves_to_known_columns(self):
        feats = ["ret_1", "ma_5", "rsi", "macd", "boll_pos", "volatility",
                 "volume_ratio", "factor_x", "macro_x"]
        groups, _ = group_feature_columns(feats)
        panel = {"groups": groups, "features": feats}
        for recipe in SUBSET_RECIPES:
            cols = _subset_columns(panel, recipe)
            assert set(cols) <= set(feats), recipe
        assert _subset_columns(panel, "full") == feats

    def test_unknown_recipe_raises(self):
        with pytest.raises(ValueError):
            _subset_columns({"groups": {}, "features": []}, "nope")

    def test_rules_are_non_empty(self):
        assert FEATURE_GROUP_RULES
        assert set(SUBSET_RECIPES) >= {"full", "compact", "returns_only"}


# ----------------------------------------------------------------------
# ② 模型族口径
# ----------------------------------------------------------------------
class TestModelFamilies:
    def test_lightgbm_adapter_uses_train_not_fit(self):
        """自研 LightGBMModel 只有 train()；适配层必须可 fit/predict_proba。"""
        m = make_model("lightgbm", _config())
        X = np.random.default_rng(0).normal(size=(60, 3))
        y = (X[:, 0] > 0).astype(float)
        m.fit(X, y)
        p = m.predict_proba(X)
        assert p.shape == (60,)
        assert np.all((p >= 0) & (p <= 1))

    def test_sklearn_family_uniform_interface(self):
        m = make_model("logistic_ridge", _config())
        X = np.random.default_rng(1).normal(size=(60, 3))
        y = (X[:, 0] + X[:, 1] > 0).astype(float)
        m.fit(X, y)
        assert m.predict_proba(X).shape == (60,)

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError):
            make_model("nope", _config())


# ----------------------------------------------------------------------
# ③ 判定表（不夸大证据）
# ----------------------------------------------------------------------
def _arm(available: bool = True, excess=None, t_exc=None, t_paired=None) -> dict:
    return {
        "available": available,
        "vs_benchmark": {"available": True, "excess_mean_per_period": excess,
                         "excess_t_stat": t_exc},
        "vs_baseline_arm": {"available": True, "diff_t_stat": t_paired},
        "verdict": {"level": "no_edge"},
    }


class TestVerdict:
    def test_positive_requires_net_excess_and_paired_t(self):
        arms = {"full": _arm(), "cand": _arm(excess=0.001, t_exc=2.5, t_paired=2.4)}
        out = _verdict_from(arms, "full", "cand", "feature_ablation")
        assert out["level"] == "positive_increment"
        assert out["winners"] == ["cand"]
        assert out["candidates"]["cand"]["increment_confirmed"] is True

    def test_positive_net_excess_but_weak_paired_t_is_inconclusive(self):
        arms = {"full": _arm(), "cand": _arm(excess=0.001, t_exc=2.5, t_paired=1.2)}
        out = _verdict_from(arms, "full", "cand", "feature_ablation")
        assert out["level"] == "inconclusive"
        assert out["candidates"]["cand"]["increment_confirmed"] is False

    def test_nonpositive_but_insignificant_is_inconclusive_not_no_increment(self):
        """净超额为负但 |t| < 2 ⇒ 没证明更好也没证明更差，不得写 no_increment。"""
        arms = {"full": _arm(), "cand": _arm(excess=-0.0005, t_exc=-1.0, t_paired=-0.4)}
        out = _verdict_from(arms, "full", "cand", "feature_ablation")
        assert out["level"] == "inconclusive"
        assert out["candidates"]["cand"]["worse_than_baseline"] is False

    def test_confirmed_worse_is_no_increment(self):
        arms = {"full": _arm(), "cand": _arm(excess=-0.003, t_exc=-2.9, t_paired=-2.5)}
        out = _verdict_from(arms, "full", "cand", "feature_ablation")
        assert out["level"] == "no_increment"
        assert out["candidates"]["cand"]["worse_than_baseline"] is True

    def test_unavailable_baseline_is_unavailable(self):
        arms = {"full": _arm(available=False)}
        out = _verdict_from(arms, "full", None, "feature_ablation")
        assert out["level"] == "unavailable"

    def test_arm_verdict_no_edge_when_excess_nonpositive(self):
        assert _arm_verdict(_arm(excess=-0.001, t_exc=-2.0))["level"] == "no_edge"

    def test_arm_verdict_positive_requires_significance(self):
        a = _arm(excess=0.002, t_exc=2.5)
        a["random_subset_control"] = {"available": True,
                                     "fraction_random_ge_signal": 0.02}
        assert _arm_verdict(a)["level"] == "positive"

    def test_arm_verdict_inconclusive_when_random_matches(self):
        a = _arm(excess=0.002, t_exc=2.5)
        a["random_subset_control"] = {"available": True,
                                     "fraction_random_ge_signal": 0.4}
        assert _arm_verdict(a)["level"] == "inconclusive"


# ----------------------------------------------------------------------
# ④ 面板：同批样本、同折、确定性
# ----------------------------------------------------------------------
class TestPanel:
    def test_panel_is_deterministic_under_symbol_shuffle(self):
        a = build_panel(_dataset(), _config(), 5, 3)
        data = _dataset()
        shuffled = {k: data[k] for k in reversed(list(data))}
        b = build_panel(shuffled, _config(), 5, 3)
        assert a["available"] and b["available"]
        assert a["n_samples"] == b["n_samples"]
        assert a["uncovered_columns"] == b["uncovered_columns"]
        assert [tuple(x[0]) for x in a["splits"]] == [tuple(x[0]) for x in b["splits"]]

    def test_panel_shares_one_split_list_for_all_arms(self):
        panel = build_panel(_dataset(), _config(), 5, 3)
        assert panel["splits"]
        # 折索引必须固定下来（臂间用同一批样本比较）
        assert all(isinstance(tr, np.ndarray) for tr, _ in panel["splits"])

    def test_panel_reports_unavailable_on_empty_data(self):
        panel = build_panel({}, _config(), 5, 3)
        assert panel["available"] is False
        assert panel["reason"]


# ----------------------------------------------------------------------
# ⑤ 端到端报告
# ----------------------------------------------------------------------
class TestBuildReport:
    def test_readonly_discipline(self):
        rep = build_report(_dataset(), _config(), horizons=(5,), folds=2,
                           n_random_controls=3, model_families=("lightgbm",),
                           feature_recipes=("full", "returns_only"))
        assert rep["affects_gate"] is False
        assert rep["readonly"] is True

    def test_empty_data_is_unavailable(self):
        rep = build_report({}, _config(), horizons=(5,))
        assert rep["available"] is False
        assert rep["reason"]

    def test_unknown_recipe_reports_reason(self):
        rep = build_report(_dataset(), _config(), horizons=(5,),
                           feature_recipes=("nope",))
        assert rep["available"] is False
        assert "未知" in rep["reason"]

    def test_bad_cost_level_reports_reason(self):
        rep = build_report(_dataset(), _config(), horizons=(5,),
                           cost_level="nope")
        assert rep["available"] is False
        assert "未知成本档" in rep["reason"]

    def test_scoring_rule_is_written_into_report(self):
        rep = build_report(_dataset(), _config(), horizons=(5,), folds=2,
                           n_random_controls=3, model_families=("lightgbm",),
                           feature_recipes=("full",))
        assert "净超额" in rep["scoring_rule"]

    def test_baseline_recipe_always_present_even_if_not_requested(self):
        """基线格必须存在（否则所有候选都没有对照）。"""
        rep = build_report(_dataset(), _config(), horizons=(5,), folds=2,
                           n_random_controls=3, model_families=("lightgbm",),
                           feature_recipes=("returns_only",))
        h = rep["per_horizon"]["5"]
        if h.get("available"):
            assert "full" in h["feature_arms"]
            assert h["baseline_arm"]["n_features"] >= 1

    def test_arms_share_identical_rebalance_period_count(self):
        """同批样本同折 ⇒ 各臂调仓期数必须一致（否则不是配对比较）。"""
        rep = build_report(_dataset(), _config(), horizons=(5,), folds=2,
                           n_random_controls=3,
                           model_families=("lightgbm", "logistic_ridge"),
                           feature_recipes=("full", "returns_only"))
        h = rep["per_horizon"]["5"]
        if not h.get("available"):
            pytest.skip("面板不可用")
        counts = {a["vs_benchmark"].get("n_periods")
                  for a in list(h["feature_arms"].values())
                  + list(h["model_arms"].values())
                  if a.get("vs_benchmark")}
        assert len(counts) == 1

    def test_no_arm_may_claim_positive_without_net_excess(self):
        """决定性纪律守卫：任何臂的 verdict 为 positive 必须净超额 > 0。"""
        rep = build_report(_dataset(), _config(), horizons=(5,), folds=2,
                           n_random_controls=5,
                           model_families=("lightgbm", "logistic_ridge"),
                           feature_recipes=("full", "returns_only"))
        h = rep["per_horizon"]["5"]
        for arm in list(h.get("feature_arms", {}).values()) + \
                list(h.get("model_arms", {}).values()):
            v = (arm or {}).get("verdict") or {}
            if v.get("level") == "positive":
                assert (arm["vs_benchmark"]["excess_mean_per_period"] or 0) > 0

    def test_verdict_levels_are_known_values(self):
        rep = build_report(_dataset(), _config(), horizons=(5,), folds=2,
                           n_random_controls=3, model_families=("lightgbm",),
                           feature_recipes=("full",))
        known = {"positive_increment", "no_increment", "inconclusive", "unavailable"}
        assert rep.get("feature_verdict_level") in known
        assert rep.get("model_verdict_level") in known

    def test_min_periods_constant_is_positive(self):
        assert MIN_PERIODS >= 8


class TestMultipleComparison:
    def test_holm_correction_is_applied_and_monotone(self):
        from src.eval.feature_model_ablation import _apply_multiple_comparison

        cands = {
            "a": {"paired_diff_t": 3.9},
            "b": {"paired_diff_t": 2.6},
            "c": {"paired_diff_t": 1.2},
            "d": {"paired_diff_t": 0.2},
            "e": {"paired_diff_t": -0.3},
        }
        _apply_multiple_comparison(cands)
        for c in cands.values():
            assert c["paired_p_raw"] is not None
            assert c["paired_p_holm"] is not None
        # Holm 后的 p 必须 ≥ 原始 p（且在 [0,1] 内）
        for c in cands.values():
            assert c["paired_p_holm"] >= c["paired_p_raw"] - 1e-9
            assert 0.0 <= c["paired_p_holm"] <= 1.0
        # 最小原始 p 的 Holm 值 = min(1, m × p)
        raw = min(c["paired_p_raw"] for c in cands.values())
        assert cands["a"]["paired_p_holm"] == pytest.approx(min(1.0, 5 * raw), abs=1e-6)

    def test_none_t_is_reported_not_faked(self):
        from src.eval.feature_model_ablation import _apply_multiple_comparison

        cands = {"a": {"paired_diff_t": None}}
        _apply_multiple_comparison(cands)
        assert cands["a"]["paired_p_raw"] is None
        assert cands["a"]["paired_p_holm"] is None
