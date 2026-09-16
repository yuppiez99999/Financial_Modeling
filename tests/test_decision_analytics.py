"""决策源有效性分析守卫（src/eval/decision_analytics.py）。

全部离线合成样本。核心是**保守判定**纪律：
  - 样本不足必须是 insufficient（不是"无效"，更不是"有效"）；
  - 判定必须**两条腿同时过**（命中率 + 平均已实现收益），
    一个"命中率高但平均收益为负"的因子必须被判 ineffective
    —— 这是本模块存在的理由：只看命中率的评估发现不了它。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import decision_analytics as da  # noqa: E402


def _rec(conf, ret, hit, horizon="short_term"):
    return {
        "verified": True,
        "probability": 0.5 + conf / 2,
        "confidence": conf,
        "actual_return": ret,
        "hit": hit,
        "symbol": "X.SH",
        "horizon": horizon,
    }


# ---------------------------------------------------------------- 样本抽取
def test_extract_skips_unverified_and_bad_rows():
    records = [
        _rec(0.4, 0.03, True),
        {**_rec(0.4, 0.03, True), "verified": False},          # 未到期 → 排除
        {**_rec(0.4, 0.03, True), "actual_return": None},       # 无收益 → 排除
        {**_rec(0.4, 0.03, True), "hit": None},                 # 无命中 → 排除
        {**_rec(0.4, 0.03, True), "confidence": None,
         "probability": None},                                  # 无置信 → 排除
    ]
    samples = da.extract_samples(records)
    assert len(samples) == 1


def test_extract_falls_back_to_proba_distance_for_confidence():
    records = [{"verified": True, "probability": 0.8, "confidence": None,
                "actual_return": 0.02, "hit": True, "horizon": "short_term"}]
    samples = da.extract_samples(records)
    assert len(samples) == 1
    assert samples[0]["confidence"] == pytest.approx(0.6)


# ---------------------------------------------------------------- 分档
def test_bin_reports_coverage_and_signed_return():
    """高置信档必须给出**带符号**的平均收益（只报命中率会漏掉涨小跌大）。"""
    samples = ([_rec(0.05, 0.01, True) for _ in range(30)]
               + [_rec(0.8, -0.05, True) for _ in range(30)])
    rows = da.by_confidence_bin(samples, min_samples=20)
    low = next(r for r in rows if r["bin"].startswith("[0.0"))
    high = next(r for r in rows if r["bin"].startswith("[0.7"))
    assert low["available"] and high["available"]
    assert low["mean_return"] > 0 > high["mean_return"]
    assert high["hit_rate"] == 1.0        # 命中率漂亮
    assert high["mean_return"] == pytest.approx(-0.05)   # 但收益为负


def test_bin_marks_insufficient_without_guessing():
    rows = da.by_confidence_bin([_rec(0.05, 0.01, True), _rec(0.9, 0.02, True)],
                                min_samples=20)
    assert all(not r["available"] for r in rows)
    assert all("reason" in r for r in rows)


# ---------------------------------------------------------------- 门槛扫描
def test_threshold_scan_coverage_is_monotone_nonincreasing():
    samples = [_rec(c, 0.01, True) for c in [0.05] * 40 + [0.4] * 30 + [0.8] * 30]
    rows = da.threshold_scan(samples, thresholds=(0.0, 0.2, 0.5), min_samples=20)
    covs = [r["coverage"] for r in rows]
    assert covs == sorted(covs, reverse=True)
    assert rows[0]["coverage"] == 1.0


# ---------------------------------------------------------------- 判定纪律
def test_verdict_insufficient_on_small_sample():
    v = da.verdict([_rec(0.5, 0.05, True) for _ in range(10)], threshold=0.2)
    assert v["status"] == da.STATUS_INSUFFICIENT
    assert "样本不足" in v["reason"]


def test_verdict_ineffective_when_high_confidence_loses_money():
    """命中率高但平均收益低于基线 → 必须判 ineffective（本模块的核心价值）。"""
    samples = ([_rec(0.05, 0.05, True) for _ in range(60)]      # 基线：低置信高收益
               + [_rec(0.8, -0.01, True) for _ in range(60)])   # 高置信：命中率100% 但亏
    v = da.verdict(samples, threshold=0.2)
    assert v["status"] == da.STATUS_INEFFECTIVE
    assert v["subset_hit_rate"] > v["baseline_hit_rate"]        # 命中率腿过了
    assert v["subset_mean_return"] < v["baseline_mean_return"]  # 收益腿没过


def test_verdict_effective_only_when_both_legs_pass():
    samples = ([_rec(0.05, 0.0, False) for _ in range(60)]
               + [_rec(0.8, 0.05, True) for _ in range(60)])
    v = da.verdict(samples, threshold=0.2)
    assert v["status"] == da.STATUS_EFFECTIVE
    assert v["subset_hit_rate"] > v["baseline_hit_rate"]
    assert v["subset_mean_return"] > v["baseline_mean_return"]


# ---------------------------------------------------------------- 汇总
def test_analyze_reports_unavailable_without_samples():
    rep = da.analyze([])
    assert rep["available"] is False
    assert rep["reason"]
    assert rep["affects_gate"] is False and rep["readonly"] is True


def test_analyze_splits_per_horizon():
    samples = ([_rec(0.8, 0.05, True, "short_term") for _ in range(70)]
               + [_rec(0.8, 0.05, True, "mid_term") for _ in range(30)])
    rep = da.analyze(samples, threshold=0.2)
    assert set(rep["per_horizon"]) == {"short_term", "mid_term"}
    assert rep["per_horizon"]["mid_term"]["n_samples"] == 30
    assert rep["n_samples"] == 100


def test_analyze_never_raises_on_garbage():
    for bad in (None, [None], [{}]):
        rep = da.analyze(bad or [])
        assert isinstance(rep, dict)
