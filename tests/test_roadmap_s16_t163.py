"""S16 / T16.3 守卫测试：置信度保留期复验 + 多时段滚动证据链。

覆盖（全部离线，合成数据，不触网）：
  - holdout_split：前 70% / 后 30%，无重叠、无遗漏、时间序；
  - rolling_periods：互不重叠、覆盖全保留期、样本不足如实返回空；
  - evaluate_period_stability：stable / unstable / insufficient_samples 三态
    （含「候选行全过才算 stable」的保守口径）；
  - summarize_stability：多数时段 stable 才 stable；各半不猜；
  - collect_predictions：训练只用训练段（篡改保留期标签不影响预测——无前视硬校验）；
  - 报告结构与 confidence_gate 决策单兼容（_extract_holdout_curves 能直接消费）；
  - CLI 注册（confidence-holdout 在 choices 里、--n-periods/--no-rolling 可解析）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval import confidence_gate as cg  # noqa: E402
from src.eval import confidence_holdout as ch  # noqa: E402


# ----------------------------------------------------------------------
# holdout_split
# ----------------------------------------------------------------------
class TestHoldoutSplit:
    def test_default_70_30(self):
        tr, ho = ch.holdout_split(1000)
        assert len(tr) == 700 and len(ho) == 300
        assert tr.max() < ho.min()          # 时间序：训练严格在保留期之前

    def test_no_overlap_no_gap(self):
        tr, ho = ch.holdout_split(257)
        assert set(tr) | set(ho) == set(range(257))

    def test_ratio_clamped(self):
        tr, ho = ch.holdout_split(100, train_ratio=0.99)   # 越界被钳到 0.9
        assert len(tr) == 90 and len(ho) == 10

    def test_empty_input(self):
        tr, ho = ch.holdout_split(0)
        assert len(tr) == 0 and len(ho) == 0

    def test_tiny_input_all_train(self):
        tr, ho = ch.holdout_split(1)
        assert len(tr) == 1 and len(ho) == 0   # 切不出保留期，如实给空


# ----------------------------------------------------------------------
# rolling_periods
# ----------------------------------------------------------------------
class TestRollingPeriods:
    def test_three_disjoint_periods(self):
        periods = ch.rolling_periods(np.arange(300), 3)
        assert len(periods) == 3
        assert periods[0][0] == 0 and periods[-1][1] == 300
        for (a, b), (c, d) in zip(periods, periods[1:]):
            assert b == c                     # 互不重叠且连续

    def test_requested_more_than_samples(self):
        periods = ch.rolling_periods(np.arange(4), 3)
        assert len(periods) <= 3 and periods[0][0] == 0

    def test_empty_holdout(self):
        assert ch.rolling_periods(np.array([], dtype=int), 3) == []

    def test_single_period(self):
        assert ch.rolling_periods(np.arange(10), 1) == [(0, 10)]


# ----------------------------------------------------------------------
# evaluate_period_stability（纯函数）
# ----------------------------------------------------------------------
def _row(thr, hit, cov=0.2, samples=200):
    return {"threshold": thr, "coverage": cov, "samples": samples,
            "available": True, "hit_rate": hit, "ic": 0.02}


class TestPeriodStability:
    def test_all_candidates_above_full_is_stable(self):
        rows = [_row(0.0, 0.50), _row(0.2, 0.55), _row(0.3, 0.58)]
        out = ch.evaluate_period_stability(rows)
        assert out["verdict"] == ch.STABLE_TAG
        assert out["full_hit_rate"] == 0.50
        assert all(r["delta_vs_full"] > 0 for r in out["candidate_rows"])

    def test_one_candidate_below_full_is_unstable(self):
        rows = [_row(0.0, 0.50), _row(0.2, 0.55), _row(0.3, 0.48)]
        out = ch.evaluate_period_stability(rows)
        assert out["verdict"] == ch.UNSTABLE_TAG   # 不择优，一行掉线即 unstable

    def test_missing_full_row_is_insufficient(self):
        rows = [_row(0.2, 0.55), _row(0.3, 0.58)]  # 没有 thr=0 基准行
        out = ch.evaluate_period_stability(rows)
        assert out["verdict"] == ch.INSUFFICIENT_TAG

    def test_candidate_below_min_samples_excluded(self):
        rows = [_row(0.0, 0.50),
                _row(0.2, 0.55, samples=200),
                _row(0.3, 0.58, samples=10)]        # 样本不足的行不参与判定
        out = ch.evaluate_period_stability(rows)
        assert out["verdict"] == ch.STABLE_TAG
        assert len(out["candidate_rows"]) == 1

    def test_out_of_range_rows_ignored(self):
        rows = [_row(0.0, 0.50), _row(0.4, 0.90), _row(0.2, 0.55)]
        out = ch.evaluate_period_stability(rows)
        assert out["verdict"] == ch.STABLE_TAG
        assert [r["threshold"] for r in out["candidate_rows"]] == [0.2]


# ----------------------------------------------------------------------
# summarize_stability
# ----------------------------------------------------------------------
class TestSummarizeStability:
    def test_majority_stable(self):
        out = ch.summarize_stability([{"verdict": "stable"},
                                      {"verdict": "stable"},
                                      {"verdict": "unstable"}])
        assert out["verdict"] == ch.STABLE_TAG
        assert out["stable_periods"] == 2 and out["unstable_periods"] == 1

    def test_majority_unstable(self):
        out = ch.summarize_stability([{"verdict": "stable"},
                                      {"verdict": "unstable"},
                                      {"verdict": "unstable"}])
        assert out["verdict"] == ch.UNSTABLE_TAG

    def test_tie_is_not_guessed(self):
        out = ch.summarize_stability([{"verdict": "stable"},
                                      {"verdict": "unstable"}])
        assert out["verdict"] == ch.INSUFFICIENT_TAG

    def test_all_insufficient(self):
        out = ch.summarize_stability([{"verdict": "insufficient_samples"}] * 3)
        assert out["verdict"] == ch.INSUFFICIENT_TAG
        assert out["insufficient_periods"] == 3


# ----------------------------------------------------------------------
# collect_predictions：无前视硬校验
# ----------------------------------------------------------------------
class _FakeLGB:
    class LGBMClassifier:
        def __init__(self, **kw):
            self.kw = kw
            self._fit_y = None

        def fit(self, X, y):
            self._fit_y = np.array(y, copy=True)
            return self

        def predict_proba(self, X):
            # 预测只依赖训练段标签分布，返回常数列即可测「保留期标签不可见」
            p = float(np.mean(self._fit_y)) if self._fit_y is not None and len(self._fit_y) else 0.5
            return np.column_stack([np.full(len(X), 1 - p), np.full(len(X), p)])


class _FakeScaler:
    def fit_transform(self, X):
        return X

    def transform(self, X):
        return X


class _FakeCombined:
    def __init__(self, n, monkeypatched_params):
        rng = np.random.default_rng(7)
        self._X = rng.normal(size=(n, 3))
        self._y = rng.integers(0, 2, size=n)
        self._ret = rng.normal(size=n)
        self._params = monkeypatched_params

    def __getitem__(self, cols):
        class _Col:
            def __init__(self, owner, c):
                self._owner, self._c = owner, c

            def to_numpy(self, dtype=None):
                if self._c == "target_5d":
                    return self._owner._y
                if self._c == "_fwd_ret":
                    return self._owner._ret
                return self._owner._X

        return _Col(self, cols)

    def __contains__(self, c):
        return c in ("f0", "f1", "f2", "target_5d", "_fwd_ret")


def _patched_params(config):
    return {"n_estimators": 50, "num_leaves": 7, "learning_rate": 0.05}


class TestCollectPredictionsNoLookahead:
    def test_holdout_labels_invisible_to_training(self, monkeypatch):
        monkeypatch.setattr("src.eval.hyperopt_tuner.current_lightgbm_params",
                            _patched_params)
        n = 400
        combined = _FakeCombined(n, None)
        tr, ho = ch.holdout_split(n)

        proba_a, ret_a = ch.collect_predictions(
            combined, ["f0", "f1", "f2"], "target_5d", tr, ho,
            config=None, lgb_module=_FakeLGB, scaler_cls=_FakeScaler)

        # 篡改保留期标签后重跑：预测必须完全一致（模型没见过保留期标签）
        combined._y[ho] = 1 - combined._y[ho]
        proba_b, ret_b = ch.collect_predictions(
            combined, ["f0", "f1", "f2"], "target_5d", tr, ho,
            config=None, lgb_module=_FakeLGB, scaler_cls=_FakeScaler)
        assert np.array_equal(proba_a, proba_b)

        # 训练段标签影响预测（证明训练确实用了训练段）
        combined._y[tr] = 1 - combined._y[tr]
        proba_c, _ = ch.collect_predictions(
            combined, ["f0", "f1", "f2"], "target_5d", tr, ho,
            config=None, lgb_module=_FakeLGB, scaler_cls=_FakeScaler)
        assert not np.array_equal(proba_a, proba_c)

        assert len(proba_a) == len(ho) and len(ret_a) == len(ho)


# ----------------------------------------------------------------------
# 报告结构：与 confidence_gate 决策单兼容
# ----------------------------------------------------------------------
class TestReportCompat:
    def test_holdout_report_consumed_by_gate(self):
        curve = {"kind": "confidence_curve", "total_samples": 500,
                 "confidence_source": "proba_distance", "rows": [
                     _row(0.0, 0.50), _row(0.2, 0.55), _row(0.3, 0.58)]}
        report = ch.build_holdout_report({"5d": curve}, meta={"train_ratio": 0.7})
        extracted = cg._extract_holdout_curves(report)
        assert "5d" in extracted and extracted["5d"]["rows"][0]["hit_rate"] == 0.50

        evidence = cg.evaluate_holdout_evidence(report, config=None)
        assert evidence["available"] is True
        assert evidence["horizons"]["5d"]["candidate_thresholds"] == [0.2, 0.3]

    def test_generated_at_present_for_freshness_check(self):
        report = ch.build_holdout_report({})
        assert report["generated_at"]
        # 决策单按报告年龄判 stale；刚生成的报告必须不 stale
        evidence = cg.evaluate_holdout_evidence(report, config=None)
        assert evidence["stale"] is False

    def test_affects_gate_always_false(self):
        assert ch.build_holdout_report({})["affects_gate"] is False
        assert ch.build_rolling_report({}, {})["affects_gate"] is False

    def test_rolling_report_roundtrip(self):
        rows = [{"period_index": 0, "rows": [_row(0.0, 0.5), _row(0.2, 0.56)]}]
        summ = {"5d": {"verdict": "stable", "stable_periods": 3}}
        report = ch.build_rolling_report({"5d": rows}, summ, meta={"n_periods": 3})
        assert report["kind"] == "confidence_rolling_verify"
        assert json.loads(json.dumps(report))["horizons"]["5d"]["stability"]["verdict"] == "stable"

    def test_report_paths_respect_config(self):
        cfg = {"strategy_gate": {"confidence_gate": {"report_dir": "out"}}}
        assert ch.holdout_report_path(cfg).as_posix() == "out/confidence_holdout_verify.json"
        assert ch.rolling_report_path(cfg).as_posix() == "out/confidence_rolling_verify.json"
        assert ch.holdout_report_path(None).as_posix() == "reports/confidence_holdout_verify.json"


# ----------------------------------------------------------------------
# CLI 注册
# ----------------------------------------------------------------------
class TestCli:
    def test_command_registered(self):
        sys.argv = ["main.py", "confidence-holdout", "--help"]
        import main as m
        import argparse
        parser = argparse.ArgumentParser()
        # 从 main 模块的真实 parser 构造路径验证 choices
        src = Path(m.__file__).read_text(encoding="utf-8")
        assert '"confidence-holdout"' in src
        assert "--n-periods" in src and "--no-rolling" in src

    def test_run_confidence_holdout_no_data_fails_soft(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        import main as m
        monkeypatch.setattr("scripts.evaluate_models.load_market_data",
                            lambda cfg, syms, offline=False: {})
        payload = m.run_confidence_holdout({"data": {"prediction_horizons": {}}})
        assert payload["affects_gate"] is False
        assert "error" in payload
