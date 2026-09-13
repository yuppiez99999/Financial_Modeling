"""S23 漂移监控守卫：PSI/KS 自研口径 + fail-close + 交叉分析边界。

全部离线合成数据。聚焦：
  1. PSI/KS 的方向性：同分布 ≈ 0 / 低漂移；平移+方差异化 → 显著漂移；
  2. 样本不足（< 30）fail-close 不猜；
  3. rolling PSI 无未来函数（参考窗固定为序列开头）；
  4. 交叉分析：相关可算 / 常数序列与样本不足如实 unavailable。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import drift_monitor as dm  # noqa: E402


def _series(n=300, seed=7, mu=0.0, sd=1.0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sd, n),
                     index=pd.date_range("2024-01-01", periods=n, freq="D"))


# ---------------------------------------------------------------- PSI / KS
def test_psi_ks_near_zero_for_same_distribution():
    s = _series(400)
    p = dm.psi(s.iloc[:200], s.iloc[200:])
    k = dm.ks_statistic(s.iloc[:200], s.iloc[200:])
    assert p is not None and p < dm.PSI_STABLE
    assert k is not None and k < 0.15


def test_psi_ks_detect_level_shift_and_scale_change():
    ref = _series(300, mu=0.0, sd=1.0)
    cur = _series(300, seed=99, mu=2.0, sd=2.0)   # 均值平移 + 方差放大
    p = dm.psi(ref, cur)
    k = dm.ks_statistic(ref, cur)
    assert p > dm.PSI_SIGNIFICANT
    assert k > dm.KS_SIGNIFICANT


def test_psi_fails_closed_below_min_sample():
    s = _series(60)
    assert dm.psi(s.iloc[:25], s.iloc[25:]) is None
    assert dm.ks_statistic(s.iloc[:25], s.iloc[25:]) is None


# ---------------------------------------------------------------- 报表
def test_drift_report_flags_shifted_feature_only():
    n = 400
    stable = _series(n, seed=1)
    shifted = pd.concat([_series(200, seed=2, mu=0.0),
                         _series(200, seed=3, mu=3.0)], ignore_index=True)
    shifted.index = stable.index
    ref_mask = pd.Series([i < 200 for i in range(n)], index=stable.index)
    cur_mask = ~ref_mask
    report = dm.drift_report({"stable": stable, "shifted": shifted},
                             ref_mask, cur_mask)
    assert report["available"] is True
    by_feat = {r["feature"]: r for r in report["rows"]}
    assert by_feat["stable"]["verdict"] == "stable"
    assert by_feat["shifted"]["verdict"] == "significant_drift"
    assert report["drifted_features"] == ["shifted"]
    assert report["affects_gate"] is False


def test_drift_report_marks_short_feature_unavailable():
    n = 100
    idx = pd.date_range("2024-01-01", periods=n)
    feats = {"ok": pd.Series(np.random.default_rng(0).normal(size=n), index=idx),
             "short": pd.Series(np.nan, index=idx)}
    ref = pd.Series([i < 50 for i in range(n)], index=idx)
    report = dm.drift_report(feats, ref, ~ref)
    by_feat = {r["feature"]: r for r in report["rows"]}
    assert by_feat["short"]["available"] is False


# ---------------------------------------------------------------- rolling
def test_rolling_psi_uses_fixed_reference_window():
    """参考窗固定为开头第一段：漂移时间线应在结构断点后抬升。"""
    flat = _series(200, seed=5, mu=0.0)
    shifted = _series(200, seed=6, mu=2.5)
    s = pd.concat([flat, shifted], ignore_index=True)
    s.index = pd.date_range("2024-01-01", periods=400, freq="D")
    tl = dm.rolling_psi(s, window=60, step=20)
    assert len(tl) >= 3
    # 前段（仍在前 200 天）应显著低于断点后
    early = tl[tl.index < pd.Timestamp("2024-08-01")]
    late = tl[tl.index >= pd.Timestamp("2024-09-01")]
    assert early.max() < late.min()


def test_rolling_psi_needs_double_window():
    assert len(dm.rolling_psi(_series(80), window=60, step=20)) == 0


# ---------------------------------------------------------------- 交叉
def test_cross_correlation_computable_and_honest():
    n = 200
    idx = pd.date_range("2024-01-01", periods=n)
    x = pd.Series(np.sin(np.arange(n) / 9.0) + np.random.default_rng(1).normal(0, .1, n), index=idx)
    y = x * 0.8 + np.random.default_rng(2).normal(0, .1, n)
    out = dm.cross_drift_with_timeline(x, y, name="demo")
    assert out["available"] is True
    assert out["pearson"] > 0.7
    assert out["affects_gate"] is False


def test_cross_correlation_rejects_constant_series():
    idx = pd.date_range("2024-01-01", periods=100)
    out = dm.cross_drift_with_timeline(
        pd.Series(np.linspace(0, 1, 100), index=idx),
        pd.Series(1.0, index=idx), name="const")
    assert out["available"] is False
    assert "常数序列" in out["reason"]
