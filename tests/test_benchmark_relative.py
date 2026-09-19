"""基准相对决策增量守卫（Issue #55 —— 回答"到底有没有跑赢什么都不做"）。

守卫点：
1. **只读纪律**：`affects_gate=False`、`readonly=True`，不改门禁 / 权重 / 池 / 配置；
2. **判据唯一**：唯一通过条件是「相对全池等权的**净超额**为正且优于随机子集」——
   绝对收益为正 / IC 为正 / 命中率 >0.5 都**不**是通过条件；
3. **含成本**：净超额扣 `2 × 单边成本`（T11.2 定稿三档，单一事实源）；
4. **非重叠**：调仓日按持有期取 `grid[::h]`，不重叠 —— 防止重叠样本假装独立；
5. **无前视**：T 日收盘建仓，赚 T→T+h；未来价格不可对齐的标的**跳过**不猜；
6. **随机子集对照**：同规模随机子集也跑一遍，若信号落在其分布内则不得外推；
7. **不吞样本**：期数不足 → `available=False` + reason，不外推。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.benchmark_relative import (
    CURRENT_WEIGHTS,
    _excess_stats,
    _horizon_returns,
    _period_stats,
    _verdict,
    build_report,
)


def _config() -> dict:
    return {"data": {"prediction_horizons": {"short_term": 5, "mid_term": 10,
                                             "long_term": 20}},
            "signal": {"horizon_weights": dict(CURRENT_WEIGHTS)}}


def _prices(n: int = 400, seed: int = 3, drift: float = 0.0004) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.015, n)))
    return pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=n, freq="B"),
        "close": close,
    })


def _dataset(n_symbols: int = 6, n: int = 400) -> dict:
    return {f"S{i}.SH": _prices(n, seed=10 + i) for i in range(n_symbols)}


def _probs(data: dict, edge: float = 0.0) -> dict:
    """构造样本外概率：`edge` 越大，信号越贴未来方向（用于造真实 edge）。"""
    out = {}
    dates = sorted({d for px in data.values() for d in px["date"]})
    for h in (5, 10, 20):
        rows = []
        for sym, px in data.items():
            p = px.set_index("date")["close"]
            fwd = p.shift(-h) / p - 1
            for d in dates:
                if d not in fwd.index or not np.isfinite(fwd.get(d, np.nan)):
                    continue
                base = 0.5 + np.clip(fwd[d] * 2.0, -0.4, 0.4) * edge
                rows.append({"date": d, "_symbol": sym, "_p": float(base)})
        out[h] = pd.DataFrame(rows)
    return out


class TestHelpers:
    def test_horizon_returns_equal_weight(self):
        px = {"A": _prices(30, seed=1), "B": _prices(30, seed=2)}
        grid = sorted(px["A"]["date"])[:5]
        r = _horizon_returns(px, grid, {t: ["A", "B"] for t in grid}, 5)
        assert len(r) == 5

    def test_horizon_returns_empty_selection_is_flat(self):
        px = {"A": _prices(30, seed=1)}
        grid = sorted(px["A"]["date"])[:5]
        r = _horizon_returns(px, grid, {t: [] for t in grid}, 5)
        assert np.allclose(r, 0.0)

    def test_horizon_returns_skips_unalignable(self):
        # 只有 A 有价格；B 缺 → 只用 A，不抛错、不补 0
        px = {"A": _prices(30, seed=1)}
        grid = sorted(px["A"]["date"])[:5]
        r = _horizon_returns(px, grid, {t: ["A", "B"] for t in grid}, 5)
        assert len(r) == 5

    def test_period_stats_refuses_short_sample(self):
        st = _period_stats(np.zeros(4), 5)
        assert st["available"] is False
        assert "不足" in st["reason"]

    def test_excess_stats_refuses_short_sample(self):
        st = _excess_stats(np.zeros(4), np.zeros(4), 0.00075, 5)
        assert st["available"] is False

    def test_excess_charges_double_cost(self):
        # 信号 == 基准 → 净超额必须恰好 = −2×单边成本
        r = np.full(20, 0.01)
        st = _excess_stats(r, r, 0.00075, 5)
        assert st["excess_mean_per_period"] == pytest.approx(-0.0015)

    def test_verdict_no_edge_when_excess_nonpositive(self):
        rep = {"benchmark_relative": {"vs_benchmark": {
            "available": True, "excess_mean_per_period": -0.0001,
            "excess_t_stat": -1.0}},
            "random_subset_control": {"available": True,
                                      "fraction_random_ge_signal": 0.5}}
        assert _verdict(rep)["level"] == "no_edge"

    def test_verdict_positive_requires_t_and_control(self):
        rep = {"benchmark_relative": {"vs_benchmark": {
            "available": True, "excess_mean_per_period": 0.01,
            "excess_t_stat": 3.0}},
            "random_subset_control": {"available": True,
                                      "fraction_random_ge_signal": 0.0}}
        assert _verdict(rep)["level"] == "positive"

    def test_verdict_inconclusive_when_control_also_matches(self):
        # 净超额为正、t 也高，但随机子集也能做到 → 不得给 positive
        rep = {"benchmark_relative": {"vs_benchmark": {
            "available": True, "excess_mean_per_period": 0.01,
            "excess_t_stat": 3.0}},
            "random_subset_control": {"available": True,
                                      "fraction_random_ge_signal": 0.5}}
        assert _verdict(rep)["level"] == "inconclusive"

    def test_verdict_inconclusive_when_t_small(self):
        rep = {"benchmark_relative": {"vs_benchmark": {
            "available": True, "excess_mean_per_period": 0.01,
            "excess_t_stat": 0.4}},
            "random_subset_control": {"available": True,
                                      "fraction_random_ge_signal": 0.0}}
        assert _verdict(rep)["level"] == "inconclusive"


class TestBuildReport:
    def test_readonly_discipline(self):
        rep = build_report(_dataset(), _probs(_dataset()), _config(),
                           n_random_controls=0)
        assert rep["affects_gate"] is False
        assert rep["readonly"] is True

    def test_empty_data_is_unavailable(self):
        rep = build_report({}, {}, _config())
        assert rep["available"] is False
        assert rep["reason"]

    def test_missing_probabilities_is_unavailable(self):
        rep = build_report(_dataset(), {5: pd.DataFrame()}, _config())
        assert rep["available"] is False

    def test_benchmark_is_equal_weight_all_pool(self):
        data = _dataset()
        rep = build_report(data, _probs(data), _config(), n_random_controls=0)
        assert rep["benchmark"]["available"] is True
        assert "26" not in rep["benchmark"]["note"]  # 6 标的测试池
        assert "等权" in rep["benchmark"]["note"]

    def test_horizon_candidates_include_current_and_only_h(self):
        data = _dataset()
        rep = build_report(data, _probs(data), _config(), n_random_controls=0)
        c = rep["horizon_candidates"]
        assert "current" in c and "equal" in c
        assert "only_5d" in c and "only_10d" in c and "only_20d" in c
        assert c["current"]["weights"] == {"5": 0.3, "10": 0.35, "20": 0.35}

    def test_random_subset_control_reports_same_size(self):
        data = _dataset()
        rep = build_report(data, _probs(data), _config(), n_random_controls=10)
        ctrl = rep["random_subset_control"]
        assert ctrl["available"] is True
        assert 1 <= ctrl["subset_size"] <= len(data)
        assert 0.0 <= ctrl["fraction_random_ge_signal"] <= 1.0

    def test_random_control_disabled_is_reported_not_faked(self):
        data = _dataset()
        rep = build_report(data, _probs(data), _config(), n_random_controls=0)
        assert rep["random_subset_control"]["available"] is False
        assert rep["random_subset_control"]["reason"]

    def test_perfect_signal_beats_benchmark(self):
        """人造强 edge 信号：净超额应显著为正（判据方向性自检）。"""
        data = _dataset(n_symbols=6, n=500)
        rep = build_report(data, _probs(data, edge=1.0), _config(),
                           holding_horizon=5, n_random_controls=20)
        vs = rep["benchmark_relative"]["vs_benchmark"]
        assert vs["available"] is True
        assert vs["excess_mean_per_period"] > 0

    def test_no_edge_signal_is_not_positive(self):
        """人造无 edge 信号（概率恒定 0.5 → 全选中 = 基准）：不得给 positive。"""
        data = _dataset(n_symbols=6, n=500)
        probs = {h: df.assign(_p=0.5) for h, df in _probs(data).items()}
        rep = build_report(data, probs, _config(), holding_horizon=5,
                           n_random_controls=20)
        assert rep["verdict"]["level"] in {"no_edge", "inconclusive"}
        assert rep["verdict"]["level"] != "positive"

    def test_deterministic_under_symbol_shuffle(self):
        data = _dataset()
        probs = _probs(data)
        r1 = build_report(data, probs, _config(), n_random_controls=8, seed=3)
        shuffled = {k: data[k] for k in reversed(list(data))}
        probs_s = {h: df.iloc[::-1].reset_index(drop=True) for h, df in probs.items()}
        r2 = build_report(shuffled, probs_s, _config(), n_random_controls=8, seed=3)
        assert (r1["benchmark_relative"]["signal_mean_per_period"]
                == r2["benchmark_relative"]["signal_mean_per_period"])

    def test_bad_cost_level_reports_reason(self):
        data = _dataset()
        rep = build_report(data, _probs(data), _config(), cost_level="nope")
        assert rep["available"] is False
        assert "未知成本档" in rep["reason"]

    def test_holding_horizon_bounds_grid(self):
        data = _dataset(n=200)
        rep = build_report(data, _probs(data), _config(), holding_horizon=50,
                           n_random_controls=0)
        # grid 太短 → 如实不可用，不外推
        if not rep["available"]:
            assert "不足" in rep["reason"]


# ----------------------------------------------------------------------
# 状态分层净超额（"增量是不是只在某个市场状态下"）
# ----------------------------------------------------------------------
from src.eval.benchmark_relative import (  # noqa: E402
    MIN_REGIME_PERIODS,
    _date_regime_labels,
    _regime_breakdown,
)


class TestRegimeBreakdown:
    def _grid(self, n=40):
        return list(pd.date_range("2021-06-01", periods=n, freq="5B"))

    def test_too_few_regimes_is_unavailable_not_faked(self):
        """只有 1 个状态可用 → 拒绝给结论，不硬凑。"""
        g = self._grid()
        rb = _regime_breakdown(g, {d: "bull" for d in g},
                               np.zeros(len(g)), np.zeros(len(g)), 0.0, 5)
        assert rb["available"] is False
        assert rb["conclusion"] == "unavailable"
        assert rb["conditional_edge_hint"] is False

    def test_single_regime_short_sample_is_flagged(self):
        """某状态期数 < MIN_REGIME_PERIODS → 该状态 available=False + reason。"""
        g = self._grid(30)
        labels = {d: ("bull" if i < 2 else "range") for i, d in enumerate(g)}
        rb = _regime_breakdown(g, labels, np.zeros(len(g)), np.zeros(len(g)), 0.0, 5)
        assert rb["by_regime"]["bull"]["available"] is False
        assert "不足" in rb["by_regime"]["bull"]["reason"]
        assert rb["by_regime"]["range"]["n_periods"] == 28

    def test_excess_charges_cost_like_global_judgement(self):
        """状态内净超额与全局同源：逐期 (sig−ben)−2×成本。"""
        g = self._grid(30)
        sig = np.full(30, 0.01)
        ben = np.zeros(30)
        cost = 0.001
        rb = _regime_breakdown(g, {d: "range" for d in g}, sig, ben, cost, 5)
        exp = 0.01 - 2 * cost
        assert rb["by_regime"]["range"]["excess_mean_per_period"] == pytest.approx(exp, abs=1e-9)

    def test_spread_and_positive_hint(self):
        """某状态净超额显著为正（t≥2）→ conditional_edge_hint=True。"""
        g = self._grid(40)
        labels = {d: ("bull" if i % 2 == 0 else "range") for i, d in enumerate(g)}
        rng = np.random.default_rng(0)
        # bull 组稳定正超额，range 组稳定负超额
        sig = np.array([0.02 if i % 2 == 0 else -0.02 for i in range(40)], dtype=float)
        sig = sig + rng.normal(0, 1e-4, 40)
        rb = _regime_breakdown(g, labels, sig, np.zeros(40), 0.0, 5)
        assert rb["available"] is True
        assert rb["conditional_edge_hint"] is True
        assert rb["conclusion"] == "conditional_hint"
        assert "bull" in rb["regimes_with_positive_edge"]
        assert rb["spread"]["best"] == "bull"
        assert rb["spread"]["best_minus_worst"] > 0

    def test_unlabeled_periods_counted_not_filled(self):
        """未标状态的调仓日如实计入 n_periods_unlabeled，不填默认状态。"""
        g = self._grid(30)
        labels = {d: "range" for d in g[:20]}   # 后 10 期无标签
        rb = _regime_breakdown(g, labels, np.zeros(30), np.zeros(30), 0.0, 5)
        assert rb["n_periods_unlabeled"] == 10
        assert rb["by_regime"]["range"]["n_periods"] == 20

    def test_breakdown_is_causal_under_future_truncation(self):
        """篡改/截断 t 之后的观测 → t 及更早的状态标签必须不变（无前视）。"""
        n = 600
        data = {f"S{i}.SH": _prices(n, seed=10 + i) for i in range(4)}
        lab_full = _date_regime_labels(data)
        assert lab_full["meta"]["available"] is True
        assert lab_full["meta"]["lookahead_prefixed"] is False
        trunc = {k: v.iloc[:-100].reset_index(drop=True) for k, v in data.items()}
        lab_trunc = _date_regime_labels(trunc)
        common = sorted(set(lab_full["labels_by_date"]) & set(lab_trunc["labels_by_date"]))
        assert len(common) > 300
        for d in common:
            assert lab_full["labels_by_date"][d] == lab_trunc["labels_by_date"][d]

    def test_mode_is_honestly_labelled(self):
        """状态口径如实标注：hmm_expanding 或 rules_fallback，不冒充。"""
        data = {f"S{i}.SH": _prices(400, seed=10 + i) for i in range(4)}
        lab = _date_regime_labels(data)
        assert lab["meta"]["mode"] in ("hmm_expanding", "rules_fallback")


class TestRegimeBreakdownInReport:
    def test_report_carries_regime_breakdown_readonly(self):
        data = _dataset(n_symbols=5, n=600)
        probs = _probs(data)
        rep = build_report(data, probs, _config(), horizons=(5, 10, 20),
                           holding_horizon=5, n_random_controls=8)
        rb = rep["regime_breakdown"]
        assert rep["regime_breakdown_enabled"] is True
        assert "labels" in rb
        # 只读纪律不受状态分层影响
        assert rep["affects_gate"] is False
        assert rep["readonly"] is True

    def test_breakdown_can_be_disabled(self):
        data = _dataset(n_symbols=5, n=600)
        probs = _probs(data)
        rep = build_report(data, probs, _config(), horizons=(5, 10, 20),
                           holding_horizon=5, n_random_controls=8,
                           regime_breakdown=False)
        assert rep["regime_breakdown_enabled"] is False
        assert rep["regime_breakdown"]["available"] is False
