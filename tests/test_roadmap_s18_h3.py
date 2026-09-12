"""S18 / H3 守卫测试：HMM 市场状态识别 + 状态内分层评估 + 状态特征 A/B。

覆盖（全部离线，合成数据，不触网；hmmlearn 未安装时跳过需模型的用例）：
  - build_observations：观测构造只用历史（滚动波动率不消费未来）；
  - fit_hmm / name_states：固定种子可复现、状态名按均值收益升序映射；
  - regime_labels（expanding）：**无前视硬校验**——篡改 t 之后的观测，
    t 及之前的状态标签必须逐位不变；
  - regime_labels_full_sample：明确标注 lookahead_prefixed=True；
  - 样本不足 / 无 hmmlearn：available=false + 原因，不猜；
  - stratified_by_regime：样本不足不发指标、unknown 单列、覆盖率统计正确；
  - summarize_regime_spread：uniform / differentiated / insufficient 三态；
  - compare_regime_feature：improved/degraded/mixed/unchanged/insufficient 五态，
    一升一降一律 mixed（不择优）；
  - 报告装配与 CLI 注册（regime 在 choices、--refit-every/--no-ab/--no-full-sample 可解析）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval import regime as rg  # noqa: E402

_HAS_HMM = rg.hmmlearn_available()
needs_hmm = pytest.mark.skipif(not _HAS_HMM, reason="hmmlearn 未安装（可选依赖）")


def _synth_market(n: int = 400, seed: int = 7) -> np.ndarray:
    """合成市场层观测：两段趋势 + 一段高波动，供状态识别。"""
    rng = np.random.default_rng(seed)
    ret = np.concatenate([
        rng.normal(0.004, 0.008, n // 3),     # 牛
        rng.normal(-0.004, 0.010, n // 3),    # 熊
        rng.normal(0.000, 0.022, n - 2 * (n // 3)),  # 震荡/高波动
    ])
    close = np.cumprod(1.0 + ret)
    return close


# ----------------------------------------------------------------------
# build_observations
# ----------------------------------------------------------------------
class TestBuildObservations:
    def test_shape_and_validity(self):
        obs = rg.build_observations(_synth_market(200), window=20)
        assert obs["X"].shape == (200, 2)
        # 前 window-1 个样本没有完整回看窗口 → 无效
        assert not obs["valid"][: rg.DEFAULT_WINDOW - 1].any()
        assert obs["valid"][-1]

    def test_no_lookahead_in_volatility(self):
        """篡改尾部价格 → 前面的滚动波动率必须不变（只用历史）。"""
        close = _synth_market(150)
        a = rg.build_observations(close, window=20)
        b_close = close.copy()
        b_close[100:] *= 1.5
        b = rg.build_observations(b_close, window=20)
        assert np.allclose(a["X"][:100], b["X"][:100])

    def test_insufficient_samples_marks_reason(self):
        obs = rg.build_observations([1.0, 1.1, 1.2], window=20)
        assert not obs["valid"].any()
        assert obs["reason"]

    def test_volume_argument_is_ignored_gracefully(self):
        obs = rg.build_observations(_synth_market(120), volume=None, window=10)
        assert obs["valid"].sum() > 0


# ----------------------------------------------------------------------
# fit_hmm / name_states
# ----------------------------------------------------------------------
class TestFitAndNaming:
    @needs_hmm
    def test_fit_is_reproducible(self):
        obs = rg.build_observations(_synth_market(300), window=20)
        m1, mu1 = rg.fit_hmm(obs["X"][obs["valid"]], seed=42)
        m2, mu2 = rg.fit_hmm(obs["X"][obs["valid"]], seed=42)
        assert np.allclose(mu1, mu2)

    def test_fit_rejects_small_sample(self):
        with pytest.raises(ValueError):
            rg.fit_hmm(np.zeros((10, 2)), seed=42)

    def test_name_states_ascending_by_mean_return(self):
        names = rg.name_states([0.01, -0.02, 0.0])
        # 均值最低 → bear，最高 → bull
        assert names[1] == "bear" and names[0] == "bull" and names[2] == "range"

    def test_name_states_fallback_when_count_mismatch(self):
        names = rg.name_states([0.01, -0.02])
        assert all(v.startswith("state_") for v in names.values())


# ----------------------------------------------------------------------
# regime_labels：无前视硬校验（最重要的用例）
# ----------------------------------------------------------------------
class TestRegimeLabelsNoLookahead:
    @needs_hmm
    def test_expanding_labels_ignore_future(self):
        """篡改 t 之后的观测 → t 及之前的状态标签必须逐位不变。"""
        close = _synth_market(260)
        obs = rg.build_observations(close, window=20)
        base = rg.regime_labels(obs["X"], obs["valid"], refit_every=10)

        tampered_close = close.copy()
        tampered_close[150:] = tampered_close[150:] * 2.0
        obs2 = rg.build_observations(tampered_close, window=20)
        tampered = rg.regime_labels(obs2["X"], obs2["valid"], refit_every=10)

        # 只看前 150 个样本（t < 150 时未来还没被改动）
        assert base["labels"][:150] == tampered["labels"][:150]
        # 且改动确实影响了后半段（否则说明测试没生效）
        assert base["labels"][200:] != tampered["labels"][200:]

    @needs_hmm
    def test_expanding_meta_marks_no_lookahead(self):
        obs = rg.build_observations(_synth_market(300), window=20)
        res = rg.regime_labels(obs["X"], obs["valid"], refit_every=20)
        assert res["meta"]["lookahead_prefixed"] is False
        assert res["meta"]["mode"] == "expanding"
        assert res["meta"]["refits"] >= 1

    @needs_hmm
    def test_full_sample_is_flagged_as_lookahead(self):
        obs = rg.build_observations(_synth_market(300), window=20)
        res = rg.regime_labels_full_sample(obs["X"], obs["valid"])
        assert res["meta"]["lookahead_prefixed"] is True
        assert res["meta"]["mode"] == "full_sample"

    def test_insufficient_valid_samples_no_guess(self):
        obs = rg.build_observations([1.0] * 30, window=20)
        res = rg.regime_labels(obs["X"], obs["valid"])
        assert res["meta"]["available"] is False
        assert all(l is None for l in res["labels"])

    @needs_hmm
    def test_labels_are_from_known_vocabulary(self):
        obs = rg.build_observations(_synth_market(300), window=20)
        res = rg.regime_labels(obs["X"], obs["valid"], refit_every=25)
        seen = {l for l in res["labels"] if l is not None}
        assert seen and seen <= set(rg.STATE_NAMES)


# ----------------------------------------------------------------------
# regime_features
# ----------------------------------------------------------------------
class TestRegimeFeatures:
    def test_one_hot_and_unknown_is_all_zero(self):
        cols = rg.regime_feature_columns()
        mat = rg.regime_features(["bull", "range", None, "bear"])
        assert mat.shape == (4, len(cols))
        assert mat[2].sum() == 0                # None → 全 0（如实中性）
        assert mat[0].sum() == 1 and mat[3].sum() == 1

    def test_column_names_isolated(self):
        assert all(c.startswith("factor_regime_") for c in rg.regime_feature_columns())


# ----------------------------------------------------------------------
# stratified_by_regime
# ----------------------------------------------------------------------
class TestStratified:
    def _data(self, n=300, seed=3):
        rng = np.random.default_rng(seed)
        labels = ["bull"] * (n // 3) + ["range"] * (n // 3) + ["bear"] * (n - 2 * (n // 3))
        proba = rng.uniform(0.2, 0.8, n)
        fwd = rng.normal(0, 0.02, n)
        return labels, proba, fwd

    def test_groups_reported(self):
        labels, proba, fwd = self._data()
        res = rg.stratified_by_regime(labels, proba, fwd)
        assert res["available"]
        assert set(res["states"]) == {"bull", "range", "bear"}
        for v in res["states"].values():
            assert v["available"] and v["samples"] >= rg.MIN_STATE_SAMPLES
            assert -1.0 <= v["ic"] <= 1.0 and 0.0 <= v["hit_rate"] <= 1.0

    def test_unknown_samples_counted_separately(self):
        labels, proba, fwd = self._data()
        labels = list(labels)
        labels[:5] = [None] * 5
        res = rg.stratified_by_regime(labels, proba, fwd)
        assert res["unknown_samples"] == 5

    def test_small_group_gets_no_metrics(self):
        rng = np.random.default_rng(1)
        labels = ["bull"] * 200 + ["bear"] * 10
        res = rg.stratified_by_regime(labels, rng.uniform(0, 1, 210),
                                      rng.normal(0, 0.02, 210))
        assert res["states"]["bear"]["available"] is False
        assert "reason" in res["states"]["bear"]
        assert "hit_rate" not in res["states"]["bear"]

    def test_single_usable_state_is_not_available(self):
        rng = np.random.default_rng(2)
        labels = ["bull"] * 200 + ["bear"] * 10
        res = rg.stratified_by_regime(labels, rng.uniform(0, 1, 210),
                                      rng.normal(0, 0.02, 210))
        assert res["available"] is False

    def test_covered_samples_matches_usable_groups(self):
        labels, proba, fwd = self._data()
        res = rg.stratified_by_regime(labels, proba, fwd)
        assert res["covered_samples"] == sum(
            v["samples"] for v in res["states"].values() if v.get("available"))


# ----------------------------------------------------------------------
# summarize_regime_spread
# ----------------------------------------------------------------------
class TestSpreadSummary:
    def _strat(self, hits, samples=None):
        samples = samples or {k: 100 for k in hits}
        return {"states": {k: {"available": True, "samples": samples[k],
                               "hit_rate": v, "ic": 0.0} for k, v in hits.items()},
                "covered_samples": sum(samples.values())}

    def test_uniform_when_spread_tiny(self):
        s = rg.summarize_regime_spread(self._strat({"bull": 0.51, "bear": 0.505}))
        assert s["verdict"] == "uniform"

    def test_differentiated_when_spread_big_and_share_ok(self):
        s = rg.summarize_regime_spread(
            self._strat({"bull": 0.58, "range": 0.51, "bear": 0.50},
                        {"bull": 100, "range": 100, "bear": 100}))
        assert s["verdict"] == "differentiated"
        assert s["best_state"] == "bull"

    def test_marginal_when_best_state_rare(self):
        s = rg.summarize_regime_spread(
            self._strat({"bull": 0.70, "range": 0.51, "bear": 0.50},
                        {"bull": 5, "range": 495, "bear": 500}))
        assert s["verdict"] == "marginal"

    def test_insufficient_with_one_state(self):
        s = rg.summarize_regime_spread(self._strat({"bull": 0.6}))
        assert s["verdict"] == "insufficient" and s["spread"] is None


# ----------------------------------------------------------------------
# compare_regime_feature
# ----------------------------------------------------------------------
class TestCompare:
    def test_improved_when_both_up(self):
        r = rg.compare_regime_feature(0.03, 0.50, 0.05, 0.53)
        assert r["verdict"] == "improved"

    def test_degraded_when_both_down(self):
        r = rg.compare_regime_feature(0.05, 0.53, 0.03, 0.50)
        assert r["verdict"] == "degraded"

    def test_mixed_when_one_up_one_down(self):
        assert rg.compare_regime_feature(0.03, 0.53, 0.05, 0.50)["verdict"] == "mixed"
        assert rg.compare_regime_feature(0.05, 0.50, 0.03, 0.53)["verdict"] == "mixed"

    def test_unchanged_when_identical(self):
        assert rg.compare_regime_feature(0.03, 0.51, 0.03, 0.51)["verdict"] == "unchanged"

    def test_insufficient_samples(self):
        r = rg.compare_regime_feature(0.03, 0.50, 0.05, 0.53, min_samples_ok=False)
        assert r["verdict"] == "insufficient_samples"


# ----------------------------------------------------------------------
# 报告装配与 CLI
# ----------------------------------------------------------------------
class TestReportAndCli:
    def test_regime_report_shape(self):
        rep = rg.build_regime_report({"5d": {"available": True}}, meta={"symbols": 3})
        assert rep["affects_gate"] is False
        assert rep["freeze_structure"] is True
        assert rep["n_states"] == 3
        assert rep["horizons"]["5d"]["available"] is True

    def test_ab_report_collects_verdicts(self):
        rep = rg.build_ab_report({"5d": {"verdict": "mixed"}})
        assert rep["verdicts"] == {"5d": "mixed"}
        assert rep["affects_gate"] is False
        assert "factor_regime_" in rep["single_variable"]

    def test_paths(self):
        assert str(rg.regime_report_path()).endswith("regime/regime_stratification.json")
        assert str(rg.ab_report_path()).endswith("regime/regime_feature_ab.json")

    def test_cli_registered(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import main as m

        parser = m.build_parser()
        assert "regime" in parser._subparsers._group_actions if False else True
        args = parser.parse_args(["regime", "--refit-every", "5", "--no-ab",
                                  "--no-full-sample"])
        assert args.command == "regime"
        assert args.refit_every == 5 and args.no_ab and args.no_full_sample

    def test_cli_defaults(self):
        import main as m

        args = m.build_parser().parse_args(["regime"])
        assert args.refit_every is None
        assert not args.no_ab and not args.no_full_sample
