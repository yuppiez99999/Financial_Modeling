"""S18 / H3 路线测试：市场状态识别 + 状态分层评估（T18.1~T18.3）。

设计要点：
  - 全部离线（合成收益序列），CI 不触网、不依赖 hmmlearn 是否安装；
  - 边界优先：不改门禁、不放大结论、不把降级包装成 HMM；
  - 无前视：状态序列逐点前向推断，只用当日及之前信息；
  - 不猜：序列过短 / 状态样本不足 / 单一状态 → available=false + 原因。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.regime import (  # noqa: E402
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_RANGE,
    _forward_filter,
    _state_semantics,
    detect_regimes,
    regime_feature_increment,
    stratified_by_regime,
)


def _regime_returns(n_bull=300, n_bear=200, n_range=300, seed=0):
    rng = np.random.default_rng(seed)
    reg = np.concatenate([np.full(n_bull, 0.002),
                          np.full(n_bear, -0.003),
                          np.full(n_range, 0.0)])
    return (reg + rng.normal(0, 0.008, reg.size)).tolist()


# ----------------------------------------------------------------------
# T18.1 状态识别（无前视 + 不冒充后端）
# ----------------------------------------------------------------------
class TestRegimeDetection:
    def test_short_series_not_guessed(self):
        d = detect_regimes([0.01] * 10)
        assert d["available"] is False and "序列太短" in d["reason"]
        assert d["states"] == []

    def test_backend_label_honest_not_hmm(self):
        """降级时必须如实标注 backend，不得冒充 HMM 结果。"""
        d = detect_regimes(_regime_returns(), backend="rules")
        assert d["available"] is True
        assert d["backend"] == "rules"
        assert "rules" in d["backend"]

    def test_hmm_unavailable_does_not_silently_fall_back(self):
        """显式要求 hmm 时若不可用，必须报不可用而非静默降级。"""
        from src.eval.regime import _try_hmmlearn

        d = detect_regimes(_regime_returns(), backend="hmm")
        if _try_hmmlearn() is None:
            assert d["available"] is False and "hmmlearn" in d["reason"]
        else:
            assert d["backend"] == "hmm"

    def test_rules_states_use_only_past(self):
        """规则口径无前视：篡改历史之后的收益不得改变更早的状态。"""
        r = _regime_returns(seed=3)
        base = detect_regimes(r[:400], backend="rules")["states"]
        tampered = list(r[:400]) + [9.9] * 100
        after = detect_regimes(tampered, backend="rules")["states"][:400]
        assert base == after

    def test_state_semantics_maps_mean_order(self):
        assert _state_semantics([-0.01, 0.0, 0.01], [0.01, 0.01, 0.01]) == [
            REGIME_BEAR, REGIME_RANGE, REGIME_BULL]

    def test_state_semantics_two_states_no_range(self):
        labels = _state_semantics([-0.01, 0.01], [0.01, 0.01])
        assert labels == [REGIME_BEAR, REGIME_BULL]

    def test_forward_filter_is_causal(self):
        """前向滤波第 t 行不得受 t 之后观测影响。"""
        ll = np.log(np.array([[0.9, 0.05], [0.05, 0.9], [0.9, 0.05]]))
        trans = np.array([[0.8, 0.2], [0.3, 0.7]])
        init = np.array([0.5, 0.5])
        full = _forward_filter(ll, trans, init)
        truncated = _forward_filter(ll[:2], trans, init)
        assert np.allclose(full[:2], truncated, atol=1e-12)

    def test_hmm_when_available_has_three_semantic_states(self):
        from src.eval.regime import _try_hmmlearn

        if _try_hmmlearn() is None:
            pytest.skip("hmmlearn 未安装")
        d = detect_regimes(_regime_returns(), backend="hmm")
        assert d["available"] is True and d["backend"] == "hmm"
        assert set(d["states"]) <= {REGIME_BULL, REGIME_BEAR, REGIME_RANGE}
        assert len(set(d["states"])) >= 2
        # 已做退化检查：状态占比不得有 0
        assert min(d["state_shares"]) > 0


# ----------------------------------------------------------------------
# T18.2 状态分层评估
# ----------------------------------------------------------------------
class TestRegimeStratified:
    def _inputs(self, n=900, seed=1):
        rng = np.random.default_rng(seed)
        scores = 0.5 + 0.3 * rng.random(n)
        fwd = 0.5 * (scores - 0.5) + rng.normal(0, 0.05, n)
        states = [REGIME_BULL] * (n // 3) + [REGIME_RANGE] * (n // 3) + \
                 [REGIME_BEAR] * (n - 2 * (n // 3))
        return scores, fwd, states

    def test_groups_cover_all_regimes_and_no_gate_change(self):
        s, f, st = self._inputs()
        res = stratified_by_regime(s, f, st)
        assert res["available"] is True and res["affects_gate"] is False
        assert {g["regime"] for g in res["groups"]} == {
            REGIME_BULL, REGIME_RANGE, REGIME_BEAR}

    def test_insufficient_group_not_guessed(self):
        s, f, st = self._inputs(n=300)
        # 把 bear 压到很少
        for i in range(len(st)):
            if st[i] == REGIME_BEAR and i > 20:
                st[i] = REGIME_RANGE
        res = stratified_by_regime(s, f, st, min_samples=30)
        bear = next(g for g in res["groups"] if g["regime"] == REGIME_BEAR)
        assert bear["available"] is False and bear["hit_rate"] is None
        assert "样本不足" in bear["reason"]

    def test_condition_conclusion_does_not_overclaim(self):
        """命中率差异明显时必须带"不得直接作为门禁证据"的限定语。"""
        n = 900
        states = [REGIME_BULL] * 300 + [REGIME_RANGE] * 300 + [REGIME_BEAR] * 300
        scores = np.array([0.9] * n)
        # 熊市真实收益与打分同向、其他状态反向 → 构造明确的命中率差
        fwd = np.array([-1.0] * 300 + [-1.0] * 300 + [1.0] * 300)
        res = stratified_by_regime(scores, fwd, states)
        if res["hit_rate_spread"] is not None and res["hit_rate_spread"] >= 0.02:
            assert "不得直接作为门禁证据" in res["condition_conclusion"]
        assert "T18.4" in res["note"]

    def test_small_spread_reported_as_no_condition(self):
        """各状态命中率完全一致时，结论必须是"未观察到条件有效性"。"""
        n = 900
        states = [REGIME_BULL] * 300 + [REGIME_RANGE] * 300 + [REGIME_BEAR] * 300
        scores = np.array([0.9] * n)
        fwd = np.array([1.0] * n)   # 每个状态命中率都 100% → spread = 0
        res = stratified_by_regime(scores, fwd, states)
        assert res["hit_rate_spread"] == 0.0
        assert "未观察到明显的状态条件有效性" in res["condition_conclusion"]

    def test_empty_input_not_guessed(self):
        res = stratified_by_regime([], [], [])
        assert res["available"] is False and res["reason"] == "无样本"


# ----------------------------------------------------------------------
# T18.3 状态作为特征的增量
# ----------------------------------------------------------------------
class TestRegimeFeatureIncrement:
    def test_zero_delta_not_claimed_improved(self):
        rng = np.random.default_rng(2)
        n = 500
        s = rng.random(n)
        f = rng.normal(0, 1, n)
        ab = regime_feature_increment(s, s.copy(), f)
        assert ab["available"] is True
        assert ab["delta_ic"] == 0.0 and ab["improved"] is False
        assert "未跑出正向增量" in ab["conclusion"]

    def test_positive_delta_flags_improved_but_not_decided(self):
        rng = np.random.default_rng(3)
        n = 600
        f = rng.normal(0, 1, n)
        base = rng.random(n)
        aug = f * 0.8 + rng.normal(0, 0.1, n)   # 更强的信号
        ab = regime_feature_increment(base, aug, f)
        assert ab["improved"] is True
        assert "人工复核" in ab["conclusion"]
        assert ab["affects_gate"] is False

    def test_empty_not_guessed(self):
        ab = regime_feature_increment([], [], [])
        assert ab["available"] is False and ab["reason"] == "无样本"
