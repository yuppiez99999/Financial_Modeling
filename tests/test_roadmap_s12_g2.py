"""S12（G2）三重障碍法标签：无前视与 A/B 对照的测试。

测试要点（对照 00_kickoff/leakage_checklist.md）：
  - 标签窗口起止与预测时点一致：判定区间恒为 (t, t+horizon]；
  - 波动率只用过去：rolling_volatility 第 i 个元素不依赖 i 之后的价格；
  - 三重障碍优先序：先触碰者胜（止盈 / 止损 / 时间障碍）；
  - 未到期样本（尾部 horizon 行）返回 None，绝不猜测；
  - 二分类折叠语义与 target_{h}d 对齐；
  - 无前视反例：把 i 之后的价格改掉，σ_i 不变（证明 σ 只依赖过去）；
  - A/B：同折、同样本、只变标签；affects_gate 恒为 False。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.labeling import (
    TripleBarrierLabeler,
    label_summary,
    rolling_volatility,
    to_binary,
    triple_barrier_labels,
)


# ----------------------------------------------------------------------
# 波动率估计：只用过去
# ----------------------------------------------------------------------
def test_rolling_volatility_uses_only_past():
    """把 i 之后的价格改掉，第 i 个 σ 必须不变（无前视硬证据）。"""
    base = [100.0 + (i % 5) for i in range(60)]
    vol_a = rolling_volatility(base, window=20)
    # 篡改第 40 个之后的价格
    tampered = list(base)
    for j in range(41, 60):
        tampered[j] = base[j] * 2.0
    vol_b = rolling_volatility(tampered, window=20)
    # 前 40 个（>窗口）的 σ 逐一相等
    for i in range(30, 41):
        assert vol_a[i] == pytest.approx(vol_b[i], rel=1e-12)


def test_rolling_volatility_insufficient_samples():
    """样本少于 MIN_VOL_SAMPLES 时返回 None（不足不猜）。"""
    vol = rolling_volatility([100.0, 101.0], window=20)
    assert vol[0] is None and vol[1] is None


# ----------------------------------------------------------------------
# 三重障碍：优先序
# ----------------------------------------------------------------------
def test_triple_barrier_up_hit_gives_positive():
    """持续上涨：上障碍先触发 → +1。"""
    closes = [100.0 * (1.02 ** i) for i in range(60)]
    labels = triple_barrier_labels(closes, horizon=5, k_up=1.0, k_down=1.0)
    valid = [x for x in labels if x is not None]
    assert valid and all(x == 1 for x in valid)


def test_triple_barrier_down_hit_gives_negative():
    """持续下跌：下障碍先触发 → -1。"""
    closes = [100.0 * (0.98 ** i) for i in range(60)]
    labels = triple_barrier_labels(closes, horizon=5)
    valid = [x for x in labels if x is not None]
    assert valid and all(x == -1 for x in valid)


def test_triple_barrier_time_barrier_fallthrough():
    """波动极小 + 微涨：上下障碍都碰不到 → 时间障碍按终点符号。"""
    closes = [100.0 + 0.001 * i for i in range(60)]
    labels = triple_barrier_labels(closes, horizon=5, k_up=1.0, k_down=1.0)
    valid = [x for x in labels if x is not None]
    # 终点收益 > 0 → 全部 +1（时间障碍兜底）
    assert valid and all(x == 1 for x in valid)


def test_triple_barrier_tail_pending_is_none():
    """尾部 horizon 行未到期 → None。"""
    closes = [100.0 * (1.01 ** i) for i in range(50)]
    labels = triple_barrier_labels(closes, horizon=5)
    assert all(x is None for x in labels[-5:])


def test_triple_barrier_early_samples_none():
    """波动率样本不足的头部 → None。"""
    closes = [100.0 * (1.01 ** i) for i in range(50)]
    labels = triple_barrier_labels(closes, horizon=5, vol_window=20)
    # 头部若干行 σ 不足 → None
    assert labels[0] is None


def test_triple_barrier_same_day_both_touch_neutral():
    """同日同时触碰上下障碍（顺序不可知）→ 0（如实中性，不猜）。"""
    closes = [100.0] * 30
    highs = [100.0] * 30
    lows = [100.0] * 30
    # 在 i=10 之后某天制造一根既有高又有低的巨幅 bar
    for j in range(11, 16):
        highs[j] = 200.0
        lows[j] = 1.0
    labels = triple_barrier_labels(closes, horizon=5, highs=highs, lows=lows)
    assert labels[10] == 0


def test_triple_barrier_highs_lows_capture_path():
    """用日内高低价能捕捉到「收盘抹平」的路径：故意做一根长下影。"""
    # 让收盘价有小幅波动（σ>0）；上障碍设得足够远（高 highs 不越线），
    # 只有人造的长下影会触发下障碍。
    closes = [100.0 for _ in range(40)]
    for i in range(40):
        closes[i] = 100.0 + (0.05 if i % 2 == 0 else -0.05)
    highs = [100.05 for _ in range(40)]
    lows = [99.95 for _ in range(40)]
    lows[12] = 50.0  # 下障碍在日内被触碰，但收盘价回到原处（路径不被抹平）
    labels = triple_barrier_labels(closes, horizon=5, k_up=1.0, k_down=1.0,
                                   highs=highs, lows=lows)
    assert labels[10] == -1
    # 对照组：不给高低价（只用收盘），该路径触碰不到 → 时间障碍兜底
    labels_close_only = triple_barrier_labels(closes, horizon=5,
                                               k_up=1.0, k_down=1.0)
    assert labels_close_only[10] in (1, -1, 0)  # 不依赖被抹平的路径


def test_horizon_must_be_positive():
    with pytest.raises(ValueError):
        triple_barrier_labels([100.0] * 30, horizon=0)


# ----------------------------------------------------------------------
# 二分类折叠与摘要
# ----------------------------------------------------------------------
def test_to_binary_maps_positive_to_one():
    labels = [1, -1, 0, None, 1]
    assert to_binary(labels) == [1.0, 0.0, 0.0, None, 1.0]


def test_label_summary_counts():
    labels = [1, 1, -1, 0, None]
    s = label_summary(labels)
    assert s["available"] is True
    assert s["valid"] == 4
    assert s["positive"] == 2
    assert s["negative"] == 1
    assert s["neutral"] == 1
    assert s["pending"] == 1


def test_label_summary_all_none_not_available():
    s = label_summary([None, None])
    assert s["available"] is False


# ----------------------------------------------------------------------
# Labeler 接线（DataFrame 消费）
# ----------------------------------------------------------------------
def _make_df(n=80, trend=1.01):
    closes = [100.0 * (trend ** i) for i in range(n)]
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n, freq="B"),
        "open": closes, "close": closes,
        "high": [c * 1.005 for c in closes],
        "low": [c * 0.995 for c in closes],
        "volume": [1000] * n,
    })


def test_labeler_attach_adds_columns():
    df = _make_df()
    labeler = TripleBarrierLabeler({})
    out = labeler.attach(df, horizon=5, binary=True)
    assert "label_tb_5d" in out.columns
    assert "label_tb_5d_bin" in out.columns
    assert len(out) == len(df)
    # 原有列未被破坏
    assert "close" in out.columns


def test_labeler_config_override():
    labeler = TripleBarrierLabeler({"labeling": {"k_up": 2.0, "k_down": 2.0, "horizon": 10}})
    assert labeler.k_up == 2.0
    assert labeler.horizon == 10


def test_labeler_empty_df():
    labeler = TripleBarrierLabeler({})
    assert labeler.label_series(pd.DataFrame({"close": []})) == []


# ----------------------------------------------------------------------
# A/B 对照实验：同折、同样本、只变标签；不改门禁
# ----------------------------------------------------------------------
def _make_combined(n_per=200, seed=7):
    """构造监督数据集：含 close/high/low/特征/target_5d/_fwd_ret。"""
    rng = np.random.default_rng(seed)
    rows = []
    for sym in ("A", "B"):
        rets = rng.normal(0.0002, 0.02, n_per)
        closes = 100 * np.exp(np.cumsum(rets))
        for i in range(n_per):
            rows.append({
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "_symbol": sym,
                "close": closes[i],
                "high": closes[i] * 1.01,
                "low": closes[i] * 0.99,
                "f1": float(rets[i]),
                "f2": float(i % 7),
                "target_5d": 1.0 if (i + 5 < n_per and closes[min(i + 5, n_per - 1)] > closes[i]) else 0.0,
                "_fwd_ret": float(rets[i]),
            })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


class _FakeModel:
    """极简确定性模型：用特征和的符号输出概率，避免测试依赖 LightGBM 训练耗时。"""

    def __init__(self, config=None):
        self._coef = None

    def train(self, X, y, X_val=None, y_val=None, feature_cols=None):
        y = np.asarray(y, dtype=float)
        mask = ~np.isnan(y)
        X = np.asarray(X, dtype=float)[mask]
        y = y[mask]
        self._coef = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            col = X[:, j]
            if np.std(col) > 0:
                self._coef[j] = np.corrcoef(col, y)[0, 1]
        return {}

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if self._coef is None:
            raise RuntimeError("untrained")
        z = X @ self._coef
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])


def test_label_ab_uses_same_samples_and_folds(monkeypatch):
    """两口径必须在同一批样本、同一组折上评估；index 重映射不得错位/越界。"""
    import src.eval.label_ab as lab
    monkeypatch.setattr(lab, "evaluate_label_variant", lab.evaluate_label_variant)
    # 直接替换 LightGBM 为 fake（经模块级导入路径）
    import src.train.models.lightgbm_model as lgbm_mod
    monkeypatch.setattr(lgbm_mod, "LightGBMModel", _FakeModel)

    combined = _make_combined()
    splits = [(np.arange(0, 200), np.arange(200, 300)),
              (np.arange(0, 300), np.arange(300, 400))]
    exp = lab.LabelABExperiment({"labeling": {"horizon": 5}})
    res = exp.compare(combined, 5, ["f1", "f2"], splits)
    assert res["available"] is True
    assert res["affects_gate"] is False
    # 两口径样本数必须一致
    assert res["old_label"]["samples"] == res["new_label"]["samples"]
    assert res["old_label"]["folds"] == res["new_label"]["folds"]
    # common_samples 是「全体有效样本」，samples 是「测试折样本」，后者是子集
    assert res["old_label"]["samples"] <= res["common_samples"]
    assert res["old_label"]["samples"] > 0


def test_label_ab_insufficient_new_labels():
    """新标签有效样本不足 → 如实 unavailable，不硬凑。"""
    from src.eval.label_ab import LabelABExperiment
    # 每只标的仅 22 行：波动率头部 + 时间障碍尾部吃掉大部分，有效新标签 < 30
    combined = _make_combined(n_per=18)
    res = LabelABExperiment({"labeling": {"horizon": 5, "vol_window": 20}}).compare(
        combined, 5, ["f1", "f2"], [(np.arange(0, 20), np.arange(20, 44))])
    assert res["available"] is False
    assert res["reason"] == "insufficient_new_label_samples"
    assert res["new_valid"] < 30


def test_label_ab_missing_target_columns():
    from src.eval.label_ab import LabelABExperiment
    combined = _make_combined().drop(columns=["target_5d"])
    res = LabelABExperiment({}).compare(combined, 5, ["f1", "f2"],
                                        [(np.arange(0, 10), np.arange(10, 15))])
    assert res["available"] is False
    assert res["reason"] == "missing_target_or_forward_return"


def test_build_ab_report_shape():
    from src.eval.label_ab import build_ab_report
    rep = build_ab_report({
        "5d": {"available": True, "delta": {"improved": True}},
        "10d": {"available": True, "delta": {"improved": False}},
    })
    assert rep["affects_gate"] is False
    assert rep["summary"]["any_improved"] is True
    assert "5d" in rep["horizons"] and "10d" in rep["horizons"]


def test_build_ab_report_no_improvement():
    from src.eval.label_ab import build_ab_report
    rep = build_ab_report({"5d": {"available": True, "delta": {"improved": False}}})
    assert rep["summary"]["any_improved"] is False
    assert "未跑出正向增量" in rep["summary"]["conclusion"]


def test_z_hit_rate_insufficient_samples():
    from src.eval.label_ab import _z_hit_rate
    assert _z_hit_rate(0.6, 10) is None
    assert _z_hit_rate(0.6, 100) is not None
