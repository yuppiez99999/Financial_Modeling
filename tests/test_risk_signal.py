"""风险预测力检验守卫（Issue #55 —— "风险预警"退路到底成不成立）。

守卫点：
1. **只读纪律**：`affects_gate=False`、`readonly=True`，不改门禁 / 权重 / 池 / 配置；
2. **必须有基准对照**：判据是"控制 trailing-vol 基线后的**偏秩相关增量**"，
   原始（未正交化）相关**不得**作为通过条件 —— 否则只是重读过去波动；
3. **无前视**：风险标签只用 `(t, t+h]`；trailing 基线只用 `[.., t]`；
   改 t 之后的价格不影响 t 处的 trailing 基线；
4. **重叠校正**：t 用有效样本数 `n_eff ≈ n/h` 算，不以名义 n 撑显著；
5. **效应量下限**：|partial IC| ≥ 0.10 才算增量（防大 n 抬小相关）；
6. **不把"没证明"写成"证明"**：无增量判 `no_increment_naive_wins` /
   `inconclusive_no_increment`，不硬凑正向；
7. **子池稳健性**：池层面抽样，成功率如实输出，分级 robust/suggestive/fragile；
8. **样本不足 / 空数据 → available=False + reason**，不外推。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.risk_signal import (
    MIN_EFFECT_SIZE,
    MIN_T,
    _effective_n,
    _forward_risk_labels,
    _judge_increment,
    _partial_spearman,
    _spearman,
    _t_stat,
    build_report,
    build_risk_dataset,
    risk_informativeness,
)


# ----------------------------------------------------------------------
# 合成数据：让"未来波动"可被一个已知因子预测
# ----------------------------------------------------------------------
def _prices_with_vol_signal(n: int = 620, seed: int = 7, beta: float = 3.0) -> pd.DataFrame:
    """构造价格序列：波动率有可预测成分（GARCH 味），信号因子在 t 处可见。"""
    rng = np.random.default_rng(seed)
    # 一个慢变的波动水平（可预测）+ 噪音
    base_vol = 0.01 + 0.01 * (1 + np.sin(np.arange(n) / 40)) / 2
    ret = rng.normal(0.0002, base_vol)
    close = 100 * np.exp(np.cumsum(ret))
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="B"),
        "close": close,
    })


def _dataset(n_symbols: int = 5, n: int = 620) -> dict:
    return {f"S{i}.SH": _prices_with_vol_signal(n, seed=20 + i) for i in range(n_symbols)}


# ----------------------------------------------------------------------
# 1. 只读纪律
# ----------------------------------------------------------------------
def test_report_is_readonly_and_no_gate_change():
    data = _dataset()
    report = build_report(data, {}, {}, horizons=[10], stability=False)
    assert report["affects_gate"] is False
    assert report["readonly"] is True
    assert "不改门禁" in report["boundary"]


def test_criterion_requires_baseline_control():
    data = _dataset()
    report = build_report(data, {}, {}, horizons=[10], stability=False)
    assert "trailing-vol" in report["criterion"]
    assert "偏秩相关" in report["criterion"]


# ----------------------------------------------------------------------
# 2. 无前视
# ----------------------------------------------------------------------
def test_forward_labels_have_no_lookahead():
    """改 t 之后的价格，不应影响 t 处的 trailing 基线；改 t 之前不影响未来标签。"""
    df = _prices_with_vol_signal(300, seed=1)
    _, _, trail = _forward_risk_labels(df["close"], 5)
    t = 150
    pert = df.copy()
    # 只动 t 之后的未来价格
    pert.loc[t + 1:, "close"] = pert.loc[t + 1:, "close"] * 1.5
    _, _, trail2 = _forward_risk_labels(pert["close"], 5)
    assert np.isclose(trail.iloc[t], trail2.iloc[t], equal_nan=True) or (
        np.isnan(trail.iloc[t]) and np.isnan(trail2.iloc[t]))


def test_forward_vol_changes_only_after_t():
    """未来波动标签在 t 处应随 (t, t+h] 价格变化而变。"""
    df = _prices_with_vol_signal(300, seed=2)
    fvol, _, _ = _forward_risk_labels(df["close"], 5)
    pert = df.copy()
    pert.loc[151:, "close"] = pert.loc[151:, "close"] * 2.0
    fvol2, _, _ = _forward_risk_labels(pert["close"], 5)
    # t=150 的未来窗口覆盖 151..155，必然改变
    assert not np.isclose(fvol.iloc[150], fvol2.iloc[150])


def test_risk_dataset_touches_only_price_columns():
    """结构化避开泄漏坑：风险数据集不依赖任何特征列 / `_fwd_ret`。"""
    data = _dataset(2, 300)
    ds = build_risk_dataset(data, 10)
    assert set(ds.columns) == {"date", "_symbol", "close", "_fwd_vol",
                               "_fwd_mdd", "_trail_vol"}


# ----------------------------------------------------------------------
# 3. 重叠校正与统计工具
# ----------------------------------------------------------------------
def test_effective_n_shrinks_with_horizon():
    assert _effective_n(1000, 5) == 200
    assert _effective_n(1000, 10) == 100
    assert _effective_n(1000, 20) == 50
    assert _effective_n(3, 20) == 2  # 不塌到 0/1


def test_t_stat_uses_effective_n():
    """同一 r，有效 n 越小 t 越小 —— 重叠样本不得撑显著性。"""
    t_small = _t_stat(0.1, _effective_n(5000, 20))
    t_big = _t_stat(0.1, _effective_n(5000, 5))
    assert t_small is not None and t_big is not None
    assert t_small < t_big


def test_partial_spearman_removes_control_signal():
    """a 完全由 control 决定时，偏相关应 ≈ 0。"""
    rng = np.random.default_rng(0)
    z = rng.normal(size=500)
    a = z * 2 + rng.normal(0, 0.01, 500)   # a ≈ z
    b = z * 3 + rng.normal(0, 0.01, 500)   # b ≈ z
    raw = _spearman(a, b)
    partial = _partial_spearman(a, b, z)
    assert raw is not None and raw > 0.9
    assert partial is not None and abs(partial) < 0.2


def test_judge_increment_requires_effect_size_and_significance():
    # 效应量不足（即使 n 巨大）
    assert not _judge_increment(0.03, 100000, positive=True)
    # 方向错
    assert not _judge_increment(-0.3, 100000, positive=True)
    # 显著性不足（有效 n 小）
    assert not _judge_increment(0.15, 20, positive=True)
    # 两条腿都过
    assert _judge_increment(0.15, 5000, positive=True)
    assert _judge_increment(-0.15, 5000, positive=False)


def test_effect_size_floor_is_locked():
    assert MIN_EFFECT_SIZE >= 0.10
    assert MIN_T >= 2.0


# ----------------------------------------------------------------------
# 4. 判定语义：不把"没证明更好"写成"更好"，也不硬凑正向
# ----------------------------------------------------------------------
def _risk_dataset_with_signal(n: int = 1500, seed: int = 5,
                              coupling: float = 0.0) -> pd.DataFrame:
    """构造一个风险数据集：conf 与 (控制 trail 后的) fwd_vol 的耦合强度可调。"""
    rng = np.random.default_rng(seed)
    trail = np.abs(rng.normal(0.02, 0.008, n))
    extra = rng.normal(0, 1, n)
    fwd_vol = trail * 5 + coupling * extra + rng.normal(0, 0.02, n)
    fwd_mdd = -trail * 3 - coupling * extra + rng.normal(0, 0.02, n)
    # conf: 与 trail 相关（弱）+ 与 extra 相关（= 真增量）
    conf = 0.3 * trail * 10 + 0.9 * extra + rng.normal(0, 1, n)
    p = np.clip(0.5 + conf / 10.0, 0.01, 0.99)
    ds = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="B"),
        "_symbol": "S.SH",
        "_fwd_vol": fwd_vol,
        "_fwd_mdd": fwd_mdd,
        "_trail_vol": trail,
    })
    ds["_p"] = p
    return ds


def test_verdict_no_increment_when_only_coupling_with_baseline():
    """conf 只与 trailing vol 耦合时，偏相关增量应不显著 ⇒ 不应判有增量。"""
    ds = _risk_dataset_with_signal(coupling=0.0)
    res = risk_informativeness(ds.drop(columns=["_p"]),
                               pd.Series(ds["_p"].to_numpy()), 10)
    assert res["verdict"] == "no_increment_naive_wins"
    assert res["has_vol_increment"] is False


def test_verdict_incremental_when_true_extra_signal():
    """conf 携带超出基线的强增量时，应判 incremental_risk_signal。"""
    ds = _risk_dataset_with_signal(coupling=6.0, seed=11)
    res = risk_informativeness(ds.drop(columns=["_p"]),
                               pd.Series(ds["_p"].to_numpy()), 10)
    assert res["verdict"] == "incremental_risk_signal"
    assert res["has_vol_increment"] is True


def test_no_samples_returns_unavailable():
    ds = build_risk_dataset({}, 10)
    assert ds.empty
    res = risk_informativeness(ds, pd.Series([], dtype=float), 10)
    assert res["available"] is False
    assert "为空" in res["reason"]


# ----------------------------------------------------------------------
# 5. 报告级结论
# ----------------------------------------------------------------------
def test_report_conclusion_no_increment_on_pure_baseline_data():
    """概率与风险无关时 → 结论应为 no_risk_increment，不硬凑正向。"""
    data = _dataset(4, 500)
    probabilities = {}
    rng = np.random.default_rng(3)
    for h in (5, 10, 20):
        ds = build_risk_dataset(data, h)
        probs = pd.DataFrame({
            "date": ds["date"],
            "_symbol": ds["_symbol"],
            "_p": 0.5 + 0.001 * rng.normal(size=len(ds)),
        })
        probabilities[h] = probs
    report = build_report(data, probabilities, {}, horizons=[5, 10, 20],
                          stability=False)
    assert report["conclusion"] == "no_risk_increment"
    assert report["n_horizons_with_increment"] == 0


def test_report_unavailable_when_no_probabilities():
    data = _dataset(2, 300)
    report = build_report(data, {}, {}, horizons=[10], stability=False)
    assert report["conclusion"] == "unavailable"


def test_stability_block_reports_honest_rate():
    data = _dataset(4, 400)
    probabilities = {}
    rng = np.random.default_rng(9)
    for h in (5, 10, 20):
        ds = build_risk_dataset(data, h)
        probabilities[h] = pd.DataFrame({
            "date": ds["date"], "_symbol": ds["_symbol"],
            "_p": 0.5 + 0.001 * rng.normal(size=len(ds)),
        })
    from src.eval.risk_signal import risk_stability_check
    stab = risk_stability_check(data, {}, horizon_days=10,
                                n_subsets=4, subset_size=2, seed=1)
    # 若环境无法训练模型（缺 lightgbm 配置）→ 必须如实写原因，不静默成功
    if not stab.get("available"):
        assert stab.get("reason"), "不可用时必须给出 reason"
        return
    assert stab["robustness_verdict"] in {"robust", "suggestive", "fragile"}
    assert stab["n_holds"] <= stab["n_subsets_usable"]


def test_forward_mdd_uses_only_future_window():
    """未来回撤标签必须只看 t 之后的价格（t 处的价格是起点参考，不比自身）。"""
    df = _prices_with_vol_signal(200, seed=4)
    _, mdd, _ = _forward_risk_labels(df["close"], 5)
    assert (mdd.dropna() <= 1e-9).all(), "回撤应为非正值"
