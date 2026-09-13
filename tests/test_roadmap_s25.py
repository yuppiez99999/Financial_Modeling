"""S25 组合级过拟合审计守卫：PSR/DSR/MinTRL/PBO（T25.1）。

全部离线合成数据（零新依赖）。聚焦：
  1. PSR 方向性：正均值 → ≈1；零均值 → ≈0.5；负均值 → ≈0；
  2. DSR < PSR（N>1 时校正更严）；N=1 退化为 PSR(SR*=0)；
  3. MinTRL：SR 越小所需跟踪期越长；SR ≤ 0 如实返回 None；
  4. PBO（CSCV 按时间块切分）：持续冠军 → 低 PBO；「前半强后半弱」的
     假冠军 → 高 PBO。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.overfit_stats import (  # noqa: E402
    audit_series,
    dsr,
    min_track_length,
    pbo_cscv,
    psr,
)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------- PSR
def test_psr_directionality():
    strong = RNG.normal(0.002, 0.005, 500)     # 年化 Sharpe ≈ 6.3
    flat = RNG.normal(0.0, 0.005, 500)
    negative = RNG.normal(-0.002, 0.005, 500)
    assert psr(strong) > 0.99
    # 零均值：单次抽样的 SR 噪声被 √n 放大，PSR 应在 0.5 附近但不至于贴边
    assert 0.05 < psr(flat) < 0.95
    assert psr(negative) < 0.05


def test_psr_insufficient_sample_or_zero_var():
    assert psr(RNG.normal(0, 1, 20)) is None                # 样本不足
    assert psr(np.zeros(100)) is None                        # 零方差不猜


# ---------------------------------------------------------------- DSR
def test_dsr_stricter_than_psr_for_multiple_trials():
    rets = RNG.normal(0.001, 0.005, 500)
    p = psr(rets)
    d10 = dsr(rets, n_trials=10)
    d100 = dsr(rets, n_trials=100)
    assert d10 < p
    assert d100 < d10            # 试验越多，校正越狠
    assert dsr(rets, n_trials=1) == pytest.approx(p)


def test_dsr_requires_positive_trials():
    rets = RNG.normal(0.001, 0.005, 500)
    with pytest.raises(ValueError):
        dsr(rets, n_trials=0)


# ---------------------------------------------------------------- MinTRL
def test_min_track_length_grows_as_sr_shrinks():
    strong = RNG.normal(0.003, 0.005, 500)
    weak = RNG.normal(0.0003, 0.005, 500)
    n_strong = min_track_length(strong)
    n_weak = min_track_length(weak)
    assert n_strong is not None and n_weak is not None
    assert n_weak > n_strong


def test_min_track_length_nonpositive_sr_is_none():
    assert min_track_length(RNG.normal(-0.001, 0.005, 500)) is None


# ---------------------------------------------------------------- PBO (CSCV)
def _matrix(persistent: bool, flip: bool = False, t=240, n=6, seed=7):
    """n 个试验：col0 = 冠军（持续强 / 前半强后半弱），其余为弱噪声。"""
    rng = np.random.default_rng(seed)
    cols = []
    if persistent:
        cols.append(rng.normal(0.003, 0.004, t))
    elif flip:
        c0 = np.concatenate([rng.normal(0.004, 0.004, t // 2),
                             rng.normal(-0.004, 0.004, t - t // 2)])
        cols.append(c0)
    for _ in range(n - 1):
        cols.append(rng.normal(0.0002, 0.004, t))
    return np.column_stack(cols)


def test_pbo_low_for_persistent_champion():
    """IS 冠军在 OS 仍是冠军 → PBO 接近 0。"""
    assert pbo_cscv(_matrix(persistent=True), n_blocks=6) < 0.2


def test_pbo_high_for_fake_champion():
    """「前半强后半弱」的假冠军：IS 选中它，OS 它垫底 → PBO 高。"""
    assert pbo_cscv(_matrix(persistent=False, flip=True), n_blocks=6) > 0.5


def test_pbo_insufficient_input_is_none():
    assert pbo_cscv(np.zeros((20, 3))) is None
    assert pbo_cscv(np.zeros((500, 1))) is None


# ---------------------------------------------------------------- 审计读数
def test_audit_series_shape():
    rets = RNG.normal(0.001, 0.005, 500).tolist()
    out = audit_series(rets, n_trials=9, name="equal/base")
    assert out["available"] is True
    assert out["psr"] is not None and out["dsr"] is not None
    assert out["dsr"] <= out["psr"]
    assert out["n_trials"] == 9
    assert out["affects_gate"] is False


def test_audit_series_short_is_unavailable():
    out = audit_series([0.001] * 10, n_trials=9)
    assert out["available"] is False
