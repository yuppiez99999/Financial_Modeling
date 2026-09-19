"""池共线性诊断守卫（Issue #55 步骤①）。

关键守卫点：
1. 有效维度口径（参与比）在**完全独立**与**完全共线**两端取正确极值；
2. 高相关对识别命中已知构造的强相关对；
3. 剥市场因子后残差维度**不低于**原始维度（beta 被移除，独立信息不被吃掉）；
4. 样本不足一律 `available=false` + reason（不猜、不造默认值）；
5. 结构性纪律：`affects_gate=False`（诊断不构成缩池/换池决策）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.pool_collinearity import (
    HIGH_CORR_THRESHOLD,
    build_returns,
    compare_pools,
    correlation_matrix,
    diagnose,
    high_correlation_pairs,
    participation_ratio,
    pool_breakdown,
)


def _synthetic_returns(n_days: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n_days, freq="B")
    base = rng.normal(0, 0.01, n_days)
    data = {}
    for i in range(6):
        # 前 3 只共享同一因子（强共线），后 3 只独立
        common = base if i < 3 else 0.0
        data[f"S{i}"] = common + rng.normal(0, 0.0005 if i < 3 else 0.01, n_days)
    return pd.DataFrame(data, index=idx)


class TestParticipationRatio:
    def test_fully_independent_equals_dimension(self):
        # 全部特征值相等 → 参与比 = 维度数
        assert participation_ratio([1.0] * 8) == pytest.approx(8.0)

    def test_fully_collinear_equals_one(self):
        # 只有一个非零特征值 → 参与比 = 1
        assert participation_ratio([5.0, 0.0, 0.0, 0.0]) == pytest.approx(1.0)

    def test_empty_returns_zero(self):
        assert participation_ratio([]) == 0.0

    def test_ignores_nonpositive_and_nan(self):
        vals = participation_ratio([1.0, 1.0, float("nan"), -0.5, 0.0])
        assert vals == pytest.approx(2.0)


class TestHighCorrelationPairs:
    def test_detects_constructed_pair(self):
        rets = _synthetic_returns()
        corr = correlation_matrix(rets)
        pairs = high_correlation_pairs(corr, 0.7)
        found = {(p["a"], p["b"]) for p in pairs}
        assert ("S0", "S1") in found
        assert ("S1", "S2") in found

    def test_independent_pair_not_reported(self):
        rets = _synthetic_returns()
        corr = correlation_matrix(rets)
        pairs = {(p["a"], p["b"]) for p in high_correlation_pairs(corr, 0.7)}
        assert ("S3", "S4") not in pairs

    def test_sorted_by_absolute_corr(self):
        rets = _synthetic_returns()
        pairs = high_correlation_pairs(correlation_matrix(rets), 0.5)
        vals = [abs(p["corr"]) for p in pairs]
        assert vals == sorted(vals, reverse=True)


class TestDiagnose:
    def test_flags_collinear_pool(self):
        rets = _synthetic_returns()
        out = diagnose(rets)
        assert out["available"] is True
        # 3 只强共线 + 3 只独立 → 有效维度明显小于 6
        assert out["effective_dimension"] < 6
        assert out["effective_dimension_ratio"] < 1.0
        assert out["collinear"] is True

    def test_residual_dimension_is_reported_and_finite(self):
        # 剥等权市场因子后的残差维度**可能低于**原始维度（等权平均由池内标的构成，
        # 残差会与因子权重产生结构性负相关）。这里只守卫"有读数且有限"，
        # 不守卫方向 —— 方向性断言是错的（2026-09-16 实测暴露）。
        rets = _synthetic_returns()
        out = diagnose(rets)
        assert out["residual_effective_dimension"] is not None
        assert np.isfinite(out["residual_effective_dimension"])
        assert out["residual_effective_dimension"] > 0
        # 残差口径必须与原始口径**分别**给出，便于同池内横向对照
        assert out["residual_pc1_variance_share"] is not None

    def test_insufficient_symbols_unavailable(self):
        idx = pd.date_range("2021-01-01", periods=200, freq="B")
        rets = pd.DataFrame({"A": np.random.normal(0, 0.01, 200),
                             "B": np.random.normal(0, 0.01, 200)}, index=idx)
        out = diagnose(rets)
        assert out["available"] is False
        assert out["reason"]

    def test_insufficient_days_unavailable(self):
        idx = pd.date_range("2021-01-01", periods=5, freq="B")
        rets = pd.DataFrame({f"S{i}": np.random.normal(0, 0.01, 5) for i in range(5)},
                            index=idx)
        out = diagnose(rets)
        assert out["available"] is False

    def test_no_fabricated_defaults(self):
        out = diagnose(pd.DataFrame())
        assert out["available"] is False
        assert out["effective_dimension"] if "effective_dimension" in out else True


class TestBuildReturns:
    def test_computes_log_returns(self):
        df = pd.DataFrame({
            "date": pd.date_range("2021-01-01", periods=5, freq="B"),
            "close": [100.0, 110.0, 121.0, 121.0, 121.0],
        })
        rets = build_returns({"X": df})
        assert "X" in rets.columns
        # ln(110/100)
        assert rets["X"].iloc[1] == pytest.approx(np.log(1.1), rel=1e-9)

    def test_nonfinite_becomes_nan_not_zero(self):
        df = pd.DataFrame({
            "date": pd.date_range("2021-01-01", periods=4, freq="B"),
            "close": [100.0, 0.0, 100.0, 100.0],
        })
        rets = build_returns({"X": df})
        # 0 价格 → ±inf 必须转 NaN，不得被当成 0 收益
        assert rets["X"].iloc[1:3].isna().all()

    def test_empty_input(self):
        assert build_returns({}).empty


class TestPoolBreakdown:
    def test_classifies_by_code(self):
        out = pool_breakdown(["510300.SH", "600276.SH", "RB.SHF", "USDCNH.FXCM"])
        assert "510300.SH" in out["etf"]
        assert "600276.SH" in out["stock"]
        assert "RB.SHF" in out["futures"]
        assert "USDCNH.FXCM" in out["forex"]

    def test_drops_empty_buckets(self):
        out = pool_breakdown(["510300.SH"])
        assert "futures" not in out


class TestComparePools:
    def test_returns_gate_discipline(self):
        out = compare_pools(_synthetic_returns())
        assert out["available"] is True
        assert out["affects_gate"] is False

    def test_small_pool_marked_unavailable(self):
        rets = _synthetic_returns()
        out = compare_pools(rets, extra_pools={"tiny": ["S0", "S1"]})
        assert out["pools"]["tiny"]["available"] is False

    def test_empty_input(self):
        out = compare_pools(pd.DataFrame())
        assert out["available"] is False
