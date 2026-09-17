"""HMM expanding 拟合路径保卫（Issue #55 第八轮）。

本轮把 `edge-check` 的状态分层从 8 标的冒烟升到**全池 38 标的**，
过程中揪出一条**早就存在**的静默路径，守卫点：

1. **起步阶段不得被误判为"拟合停滞"**：`regime_labels` 从第 1 个有效样本起
   就尝试 fit，而 `fit_hmm` 要求 `MIN_FIT_SAMPLES` —— 若不跳过前
   `MIN_FIT_SAMPLES - 1` 个样本，从头 60 次 `insufficient_fit_samples` 会被
   误计成"连续失败"，导致一个**可 fit 的序列被标成 stalled**（本轮真事故）；
2. **刻意不可 fit 的序列要如实停下**：样本量够但拟合始终退化时，
   在 `HMM_MAX_CONSECUTIVE_FAILURES` 次尝试后停止，并如实报
   `stalled=True` / `refits` 计数，**不静默空转**；
3. **状态标签失败必须 fail-soft 且不冒充**：`_date_regime_labels` 在
   `regime_labels` 抛异常时降级到规则口径并标 `mode=rules_fallback`，
   同时把异常类型写进 `hmm_meta.reason`，不静默、不假称 HMM；
4. **趋势 / 盘整二分**：bull+bear → trending、range → choppy，
   与三分**同源**（同一批逐期净超额），三分出现"牛态无期数"时该问仍有读数；
5. 全部读数仍然 `affects_gate=False`（只读）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.eval.regime as rg
from src.eval.benchmark_relative import (
    _binary_regime_breakdown,
    _date_regime_labels,
    build_report,
)


def _market_frames(n_symbols: int = 4, n: int = 420) -> dict:
    """合成一组价格表（收益有波动、无趋势），供状态拟合。"""
    out = {}
    for i in range(n_symbols):
        rng = np.random.default_rng(500 + i)
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
        out[f"S{i}.SH"] = pd.DataFrame({
            "date": pd.date_range("2021-01-01", periods=n, freq="B"),
            "close": close,
        })
    return out


def _config() -> dict:
    return {"data": {"prediction_horizons": {"short": 5, "mid": 10, "long": 20}},
            "signal": {"horizon_weights": {"5": 0.3, "10": 0.35, "20": 0.35}}}


class TestStartupSkipIsNotAStall:
    """起步阶段样本不足**不得**被当成拟合停滞（本轮真事故的回归守卫）。"""

    def test_startup_insufficient_samples_not_counted_as_failures(self):
        frames = _market_frames()
        close = frames["S0.SH"]["close"].to_numpy(dtype=float)
        obs = rg.build_observations(close, window=20)
        res = rg.regime_labels(obs["X"], obs["valid"], refit_every=20)
        meta = res["meta"]
        if not meta.get("available"):
            pytest.skip("该合成序列在 expanding 口径下不可 fit（环境差异）")
        # 关键：起步阶段的 insufficient_fit_samples 不得把 stalled 打开
        assert meta["stalled"] is False
        # 且**至少**成功 fit 过一次（否则等于从来没跑起来）
        assert meta["refits"] >= 1

    def test_no_wasted_fit_attempts_on_startup(self):
        """首个被尝试的前缀样本数 ≥ MIN_FIT_SAMPLES（不再白跑前 59 次）。

        行为守卫：用一个能 fit 的序列跑一遍，`fit_failures` 必须远小于
        护栏阈值 —— 起步阶段那次必然的 `insufficient_fit_samples` 已不再
        计入，否则 60 次起步失败就足以把 stalled 打开（本轮真事故）。
        """
        frames = _market_frames()
        close = frames["S0.SH"]["close"].to_numpy(dtype=float)
        obs = rg.build_observations(close, window=20)
        res = rg.regime_labels(obs["X"], obs["valid"], refit_every=20)
        meta = res["meta"]
        assert meta.get("fit_failures", 0) < int(rg.MIN_FIT_SAMPLES)
        assert meta.get("stalled") is False


class TestDeliberatelyUnfittableStopsHonestly:
    """刻意造不可 fit 的序列：必须停下并如实上报，不空转。"""

    def test_capped_attempts_when_never_fittable(self, monkeypatch):
        """永不退化地抛 ValueError 时：尝试次数有上限，且如实报 stalled。"""
        calls = {"n": 0}

        def _always_bad(*a, **k):
            calls["n"] += 1
            raise ValueError("always_degenerate")

        monkeypatch.setattr(rg, "fit_hmm", _always_bad)
        n = 2000
        X = np.zeros((n, 2), dtype=float)
        res = rg.regime_labels(X, np.ones(n, dtype=bool), refit_every=20)
        meta = res["meta"]
        assert meta["available"] is False
        assert meta["refits"] == 0
        # 有上限：不能每个有效样本都再试一次
        assert calls["n"] == int(rg.HMM_MAX_CONSECUTIVE_FAILURES)
        assert meta["stalled"] is True

    def test_stall_cap_is_at_least_min_fit_samples(self):
        """护栏阈值必须 ≥ MIN_FIT_SAMPLES，否则可 fit 的序列会被误停。"""
        assert int(rg.HMM_MAX_CONSECUTIVE_FAILURES) >= int(rg.MIN_FIT_SAMPLES)


class TestLabelsFailureIsFailSoftAndHonest:
    """状态标签异常 → 降级规则口径，如实标注，不拖垮主读数。"""

    def test_exception_falls_back_to_rules_and_labels_it(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("boom-for-test")

        monkeypatch.setattr(rg, "regime_labels", _boom)
        lab = _date_regime_labels(_market_frames())
        assert lab["meta"]["mode"] == "rules_fallback"
        assert "boom-for-test" in (lab["meta"].get("hmm_meta") or {}).get("reason", "")
        # 规则口径仍然给出可用标签（无前视）
        assert lab["meta"]["available"] is True

    def test_hmm_meta_is_surfaced(self):
        lab = _date_regime_labels(_market_frames())
        meta = lab["meta"]
        assert meta["mode"] in ("hmm_expanding", "rules_fallback")
        assert "hmm_meta" in meta


class TestBinaryTrendChoppy:
    """趋势 / 盘整二分：与三分同源，且必然可比。"""

    def _grid(self, n: int = 60) -> list:
        return list(pd.date_range("2021-01-01", periods=n, freq="B"))

    def test_bull_and_bear_merge_into_trending(self):
        g = self._grid(60)
        labels = {d: ("bull" if i % 3 == 0 else ("bear" if i % 3 == 1 else "range"))
                  for i, d in enumerate(g)}
        sig = np.full(len(g), 0.01)
        ben = np.zeros(len(g))
        out = _binary_regime_breakdown(g, labels, sig, ben, 0.0, 5)
        # trending = bull ∪ bear 的期数
        n_bull = sum(1 for v in labels.values() if v == "bull")
        n_bear = sum(1 for v in labels.values() if v == "bear")
        assert out["by_regime"]["trending"]["n_periods"] == n_bull + n_bear
        assert out["by_regime"]["choppy"]["n_periods"] == len(g) - n_bull - n_bear
        assert out["mapping"]["trending"] == ["bull", "bear"]

    def test_binary_available_when_ternary_bull_missing(self):
        """三分里牛态无期数时，二分仍必须给出 trending/choppy 两态读数。"""
        g = self._grid(40)
        labels = {d: ("range" if i % 2 == 0 else "bear") for i, d in enumerate(g)}
        out = _binary_regime_breakdown(g, labels, np.zeros(len(g)),
                                       np.zeros(len(g)), 0.0, 5)
        assert out["available"] is True
        assert set(out["by_regime"]) == {"trending", "choppy"}

    def test_binary_same_judgement_as_ternary(self):
        """判据同源：均匀分段时 trending 均值 == bull/bear 期数加权均值。"""
        g = self._grid(90)
        labels = {d: ("bull" if i % 3 == 0 else ("bear" if i % 3 == 1 else "range"))
                  for i, d in enumerate(g)}
        excess = np.linspace(-0.01, 0.01, len(g))
        out = _binary_regime_breakdown(g, labels, excess, np.zeros(len(g)), 0.0, 5)
        vals = [excess[i] for i, d in enumerate(g)
                if labels[d] in ("bull", "bear")]
        assert out["by_regime"]["trending"]["excess_mean_per_period"] == pytest.approx(
            float(np.mean(vals)), abs=1e-6)

    def test_binary_short_states_flagged(self):
        g = self._grid(20)
        labels = {d: ("bull" if i < 3 else "range") for i, d in enumerate(g)}
        out = _binary_regime_breakdown(g, labels, np.zeros(len(g)),
                                       np.zeros(len(g)), 0.0, 5)
        assert out["by_regime"]["trending"]["available"] is False
        assert "不足" in out["by_regime"]["trending"]["reason"]


class TestReportCarriesBinaryReadonly:
    def test_report_has_binary_and_stays_readonly(self):
        import numpy as np
        frames = _market_frames(n_symbols=5, n=600)
        dates = sorted({d for px in frames.values() for d in px["date"]})
        probs = {}
        for h in (5, 10, 20):
            rows = [{"date": d, "_symbol": s, "_p": 0.5}
                    for s in frames for d in dates]
            probs[h] = pd.DataFrame(rows)
        rep = build_report(frames, probs, _config(), horizons=(5, 10, 20),
                           holding_horizon=5, n_random_controls=8)
        rb = rep["regime_breakdown"]
        assert rb["available"] is True
        assert rb.get("regime_order")
        assert "binary" in rb
        assert rep["affects_gate"] is False
        assert rep["readonly"] is True
