"""风险预测力检验（Issue #55 —— 方向预测到顶后，"风险预警"这条退路到底成不成立）。

## 问题从哪来

Issue #55 共八轮排查（池 / 标签 / 周期 / 置信度语义 / 基准相对净超额 /
特征集 × 模型族 / 状态分层）**全部有读数、全部为负**：
现行口径相对全池等权在三个周期上**统计显著为负**（全池 t = −3.62）。
方向预测这条线**到顶**，历轮的一致建议是「另立项目做波动 / 回撤预警」。

但那条建议的**唯一地基**是一个从未独立验证过的连带结论：

> 「置信度 = 波动率探测器」⇒ 模型输出可作风险预警。

而这个地基本身在文献里是**自相矛盾**的：`regime-conditioned-signal` 实测
`IC(置信度, 波动)` 在 5 只小池上 +0.10~+0.34，在 26 只池上**接近 0**；
同一份文档在结论段却写成「置信度是波动率探测器」。**池是实验条件的一部分**，
小池的强读数不能当大池结论 —— 这正是必须做子池稳健性的原因。

⇒ 本模块把这条地基**单独拿出来、用独立判据检验**，再决定「风险预警」这条退路
是否值得立项。**不先验地相信**小池读数，也不把"建议"当"结论"。

## 判据（写死，防自由度回流）

方向预测那轮踩过的最大坑是**判据错位**（IC / 命中率 / 绝对收益 → 换判据后结论相反）。
本模块沿用同一纪律：

1. **必须有基准对照**：风险预测的"什么都不做"对照 = **用截至 t 的已实现波动率
   直接外推**（trailing vol，`_vol`）。这是波动预测的朴素强基线（波动有长记忆）。
   模型输出必须**在它之上**提供增量，才算携带风险信息 —— 否则只是"重读过去波动"。
2. **无前视**：风险标签（未来波动 / 未来最大回撤）只用 `(t, t+h]` 的价格；
   信号与对照只用 `[.., t]`。有守卫。
3. **相对增量的判据 = 正交化后的偏相关（partial IC）**：把信号与对照都对
   `trailing vol` 的秩做残差化，再看信号残差与未来风险的相关 —— 这样量到的
   才是**超出朴素基线的增量**，不是共有的"波动长记忆"。
4. **子池稳健性**：池层面抽样（不重采样样本），成功率如实输出，分级
   `robust` / `suggestive` / `fragile`。
5. **不把"没证明"写成"证明"**：样本不足 / 方向不稳 → 如实标 `inconclusive`。

## 输出纪律

只报数，`affects_gate=False`，**不**改门禁 / 权重 / 池 / 配置。
本模块回答的是「值不值得立项」，**不是**「新口径长什么样」。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_SAMPLES = 100          # 逐周期单读数下限
MIN_SUBSET_SAMPLES = 200   # 子池读数下限
TRADING_DAYS_PER_YEAR = 252
MIN_EFFECT_SIZE = 0.10     # 偏相关**效应量**下限（防"大 n 把小相关抬成显著"）
MIN_T = 2.0                # 显著性下限（用**重叠校正后**的有效 n 算）


# ----------------------------------------------------------------------
# 基础统计
# ----------------------------------------------------------------------
def _finite(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _round(v: Any, nd: int = 6) -> Optional[float]:
    f = _finite(v)
    return None if f is None else round(f, nd)


def _rank(a: Sequence[float]) -> np.ndarray:
    """秩（平均值秩处理并列）—— Spearman 的基础，抗离群。"""
    arr = np.asarray(a, dtype=float)
    order = arr.argsort(kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
    # 并列取平均秩
    s = pd.Series(arr)
    if s.duplicated().any():
        ranks = pd.Series(arr).rank(method="average").to_numpy(dtype=float)
    return ranks


def _spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 8:
        return None
    xr, yr = _rank(x[m]), _rank(y[m])
    if xr.std() < 1e-12 or yr.std() < 1e-12:
        return None
    return float(np.corrcoef(xr, yr)[0, 1])


def _partial_spearman(a: Sequence[float], b: Sequence[float],
                      control: Sequence[float]) -> Optional[float]:
    """a 与 b 在**控制** control 之后的偏秩相关（Spearman partial）。

    做法：三方都取秩 → 把 rank(a)、rank(b) 分别对 rank(control) 做最小二乘残差
    （含截距）→ 残差的 Pearson 相关。这是"超出 control 的增量"的直接读数。
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    z = np.asarray(control, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if int(m.sum()) < 20:
        return None
    xr, yr, zr = _rank(x[m]), _rank(y[m]), _rank(z[m])
    if zr.std() < 1e-12:
        return None
    zc = zr - zr.mean()
    denom = float(zc @ zc)
    if denom < 1e-12:
        return None
    rx = xr - xr.mean() - (zc @ (xr - xr.mean()) / denom) * zc
    ry = yr - yr.mean() - (zc @ (yr - yr.mean()) / denom) * zc
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _t_stat(ic: Optional[float], n: int) -> Optional[float]:
    """相关系数的朴素 t（H0: rho=0）：t = r * sqrt((n-2)/(1-r^2))。

    **注意**：这里的 n 必须是**有效独立样本数**，不是名义样本数 ——
    重叠持有期（相邻样本共享 h−1 日）会把 t 虚高约 sqrt(h) 倍。
    调用方请传 `_effective_n` 的结果。
    """
    r = _finite(ic)
    if r is None or n <= 2 or abs(r) >= 1.0:
        return None
    return float(r * np.sqrt((n - 2) / max(1e-12, 1.0 - r * r)))


def _effective_n(n: int, horizon_days: int) -> int:
    """重叠样本的有效独立样本数近似：`n_eff ≈ n / h`。

    未来 h 日标签在相邻日之间高度重叠（共享 h−1 日收益），名义 n 会
    严重高估独立性。保守按 1/h 折算 —— 与 `benchmark_relative` 的
    "按持有期取非重叠调仓日"同一纪律，只是这里不丢样本、只折算自由度。
    """
    h = max(1, int(horizon_days))
    return max(2, int(n) // h)


def _judge_increment(ic: Optional[float], n_eff: int, positive: bool) -> bool:
    """增量判定：**效应量**与**显著性**两条腿都过，方向要对。

    - 效应量：|partial IC| ≥ MIN_EFFECT_SIZE（防大 n 抬小相关）
    - 显著性：|t| ≥ MIN_T（t 用有效 n 算，已折算重叠）
    - 方向：`positive=True` 要求 IC > 0；`False` 要求 IC < 0
    """
    r = _finite(ic)
    if r is None:
        return False
    if positive and r <= 0:
        return False
    if (not positive) and r >= 0:
        return False
    if abs(r) < MIN_EFFECT_SIZE:
        return False
    t = _t_stat(r, n_eff)
    return t is not None and abs(t) >= MIN_T


# ----------------------------------------------------------------------
# 未来风险标签（只用 (t, t+h]，无前视）
# ----------------------------------------------------------------------
def _forward_risk_labels(close: pd.Series, horizon_days: int,
                         vol_window: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """逐标的：未来已实现波动率 + 未来最大回撤 + 事前 trailing 波动率。

    - `_fwd_vol`：`(t, t+h]` 上日收益的标准差 × sqrt(h)（未来已实现波动，标签）
    - `_fwd_mdd`：`(t, t+h]` 上相对区间高点的最大回撤（负值，越深越负）
    - `_trail_vol`：截至 t 的 trailing 波动率 × sqrt(h)（朴素对照基线，**无前视**）

    坑（contains）：`_fwd_mdd` 若从 t 本身算起，会把 t 处已知信息混进"未来"；
    区间必须从 `t+1` 起。这里用 `shift(-1)` 后再 rolling，保证只看 t 之后。
    """
    c = pd.to_numeric(close, errors="coerce").astype(float)
    ret1 = c.pct_change()
    trail = ret1.rolling(int(vol_window)).std() * np.sqrt(horizon_days)
    # 未来 h 日：t+1 .. t+h
    fwd_slice_ret = ret1.shift(-1)
    fwd_vol = (fwd_slice_ret.rolling(int(horizon_days)).std()
               * np.sqrt(horizon_days)).shift(-(int(horizon_days) - 1))
    # 未来最大回撤：从 t 起向前看 h+1 个价格，取路径内 (min/max_runup - 1)
    win = int(horizon_days) + 1
    roll_max = c.shift(-1).rolling(win, min_periods=win).max()
    roll_min = c.shift(-1).rolling(win, min_periods=win).min()
    # 以 t 收盘为起点，最深回撤 = min(未来价格)/起点 - 1（近似，方向敏感）
    fwd_mdd = (roll_min / c) - 1.0
    fwd_mdd = fwd_mdd.shift(-(win - 1))
    return fwd_vol, fwd_mdd, trail


def build_risk_dataset(data: Dict[str, pd.DataFrame], horizon_days: int,
                       vol_window: int = 20) -> pd.DataFrame:
    """逐标的拼风险数据集：只用价格表，**不碰特征列**（结构化避开 `_fwd_ret` 泄漏坑）。"""
    parts: List[pd.DataFrame] = []
    for symbol, px in (data or {}).items():
        if "date" not in px.columns or "close" not in px.columns:
            continue
        d = px.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        fwd_vol, fwd_mdd, trail = _forward_risk_labels(
            d["close"], int(horizon_days), int(vol_window))
        part = pd.DataFrame({
            "date": d["date"],
            "_symbol": symbol,
            "close": pd.to_numeric(d["close"], errors="coerce"),
            "_fwd_vol": fwd_vol,
            "_fwd_mdd": fwd_mdd,
            "_trail_vol": trail,
        })
        part = part.dropna(subset=["_fwd_vol", "_fwd_mdd", "_trail_vol"])
        if not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = (out.sort_values(["date", "_symbol"], kind="mergesort")
           .reset_index(drop=True))
    return out


# ----------------------------------------------------------------------
# 核心检验：模型输出对未来风险有没有**超出朴素基线**的增量
# ----------------------------------------------------------------------
def risk_informativeness(dataset: pd.DataFrame, probabilities: pd.Series,
                         horizon_days: int) -> Dict[str, Any]:
    """逐周期检验：`_p` / 置信度 对未来波动、未来回撤的增量预测力。

    `probabilities` 以 (date, _symbol) 对齐（调用方保证同源 walk-forward）。
    """
    out: Dict[str, Any] = {
        "horizon_days": int(horizon_days),
        "available": False,
        "reason": "",
    }
    if dataset is None or dataset.empty:
        out["reason"] = "风险数据集为空"
        return out

    d = dataset.copy()
    d["_p"] = probabilities.reindex(d.index).to_numpy(dtype=float)
    d = d.dropna(subset=["_p"])
    n = int(len(d))
    out["n_samples"] = n
    if n < MIN_SAMPLES:
        out["reason"] = f"样本不足（{n} < {MIN_SAMPLES}）"
        return out

    n_eff = _effective_n(n, int(horizon_days))
    out["n_effective"] = int(n_eff)
    p = d["_p"].to_numpy(dtype=float)
    conf = np.abs(p - 0.5) * 2.0
    fwd_vol = d["_fwd_vol"].to_numpy(dtype=float)
    fwd_mdd = d["_fwd_mdd"].to_numpy(dtype=float)
    trail = d["_trail_vol"].to_numpy(dtype=float)

    out.update({
        "available": True,
        "interpretation": "",
        # 朴素基线自身的强度（对照）
        "ic_trailvol_vs_fwdvol": _round(_spearman(trail, fwd_vol)),
        "ic_trailvol_vs_fwdmdd": _round(_spearman(trail, fwd_mdd)),
        # 原始读数（含共有成分）—— 只作对照，**不作判据**
        "ic_prob_vs_fwdvol": _round(_spearman(p, fwd_vol)),
        "ic_conf_vs_fwdvol": _round(_spearman(conf, fwd_vol)),
        "ic_prob_vs_fwdmdd": _round(_spearman(p, fwd_mdd)),
        "ic_conf_vs_fwdmdd": _round(_spearman(conf, fwd_mdd)),
        # 判据：控制 trailing vol 之后的**偏相关**（增量）
        "partial_ic_prob_fwdvol": _round(_partial_spearman(p, fwd_vol, trail)),
        "partial_ic_conf_fwdvol": _round(_partial_spearman(conf, fwd_vol, trail)),
        "partial_ic_prob_fwdmdd": _round(_partial_spearman(p, fwd_mdd, trail)),
        "partial_ic_conf_fwdmdd": _round(_partial_spearman(conf, fwd_mdd, trail)),
    })

    # 增量 t 值：**用重叠校正后的有效 n**（名义 n 会把 t 虚高约 sqrt(h) 倍）
    for key in ("partial_ic_prob_fwdvol", "partial_ic_conf_fwdvol",
                "partial_ic_prob_fwdmdd", "partial_ic_conf_fwdmdd"):
        out[f"{key}_t"] = _round(_t_stat(out.get(key), n_eff), 4)

    # 判定：风险预警价值 = 至少一条增量读数**效应量够 + 显著 + 方向对**
    # 方向：conf → 未来波动 正（风险大 ⇒ 波动大）；
    #       conf → 未来回撤 **负**（回撤是负值，越深越负 ⇒ 风险大相关为负）
    vol_keys = ("partial_ic_prob_fwdvol", "partial_ic_conf_fwdvol")
    mdd_keys = ("partial_ic_prob_fwdmdd", "partial_ic_conf_fwdmdd")
    vol_hits = [k for k in vol_keys
                if _judge_increment(out.get(k), n_eff, positive=True)]
    mdd_hits = [k for k in mdd_keys
                if _judge_increment(out.get(k), n_eff, positive=False)]
    out["vol_increment_hits"] = vol_hits
    out["mdd_increment_hits"] = mdd_hits
    out["has_vol_increment"] = bool(vol_hits)
    out["has_drawdown_increment"] = bool(mdd_hits)

    baseline_ic = out.get("ic_trailvol_vs_fwdvol")
    if out["has_vol_increment"] or out["has_drawdown_increment"]:
        out["verdict"] = "incremental_risk_signal"
        out["interpretation"] = (
            "模型输出在**朴素波动基线之上**仍有显著增量 ⇒ 风险预警这条退路"
            "**有地基**，值得进一步做子池稳健性确认。"
        )
    elif baseline_ic is not None and abs(baseline_ic) > 0.3:
        out["verdict"] = "no_increment_naive_wins"
        out["interpretation"] = (
            f"朴素 trailing-vol 基线本身最强（IC={baseline_ic}），模型输出"
            "**不提供超出它的增量** ⇒ 风险预警用朴素基线即可，模型无附加价值。"
        )
    else:
        out["verdict"] = "inconclusive_no_increment"
        out["interpretation"] = (
            "模型输出相对朴素基线**无显著增量**、基线自身也不强 ⇒ "
            "现有信号不足以支撑风险预警，如实记为不确定。"
        )
    return out


# ----------------------------------------------------------------------
# 子池稳健性（池层面抽样，不重采样样本）
# ----------------------------------------------------------------------
def risk_stability_check(data: Dict[str, pd.DataFrame], config: Dict[str, Any],
                         horizon_days: int = 10, folds: int = 3,
                         n_subsets: int = 12, subset_size: int = 18,
                         seed: int = 0) -> Dict[str, Any]:
    """随机子池重跑，检验「增量风险信号」是否池稳健。"""
    import random as _random

    from src.eval.regime_conditioned_signal import (
        _build_supervised, _feature_columns, _walk_forward_proba)

    out: Dict[str, Any] = {
        "kind": "risk_stability",
        "horizon_days": int(horizon_days),
        "n_subsets_requested": int(n_subsets),
        "subset_size": int(subset_size),
        "available": False,
        "reason": "",
        "trials": [],
    }
    symbols = sorted((data or {}).keys())
    if len(symbols) < subset_size or subset_size < 2:
        out["reason"] = (f"可用标的不足（{len(symbols)} < {subset_size}），"
                         "无法做子池稳健性检验")
        return out

    rng = _random.Random(int(seed))
    holds = 0
    usable = 0
    for _ in range(int(n_subsets)):
        picked = rng.sample(symbols, int(subset_size))
        sub_data = {k: data[k] for k in picked}
        try:
            sup = _build_supervised(sub_data, config, int(horizon_days))
            if sup.empty:
                continue
            feats = _feature_columns(sup, config, int(horizon_days))
            proba = _walk_forward_proba(sup, feats, int(horizon_days), config, folds)
            mask = np.isfinite(proba)
            sup = sup[mask].reset_index(drop=True)
            if len(sup) < MIN_SUBSET_SAMPLES:
                out["trials"].append({"available": False,
                                      "reason": "样本不足"})
                continue
            risk_ds = build_risk_dataset(sub_data, int(horizon_days))
            if risk_ds.empty:
                out["trials"].append({"available": False,
                                      "reason": "风险数据集为空"})
                continue
            # 对齐：以 (date, _symbol) 为键
            key_sup = sup[["date", "_symbol"]].copy()
            key_sup["_p"] = proba[mask]
            key_sup["date"] = pd.to_datetime(key_sup["date"]).dt.normalize()
            risk_ds["date"] = pd.to_datetime(risk_ds["date"]).dt.normalize()
            merged = risk_ds.merge(key_sup, on=["date", "_symbol"], how="inner")
            if len(merged) < MIN_SUBSET_SAMPLES:
                out["trials"].append({"available": False,
                                      "reason": "合并后样本不足"})
                continue
            res = _risk_core(merged, int(horizon_days))
            usable += 1
            holds += int(bool(res.get("holds")))
            out["trials"].append({
                "available": True,
                "n_samples": int(res.get("n_samples", 0)),
                "partial_ic_conf_fwdvol": res.get("partial_ic_conf_fwdvol"),
                "partial_ic_conf_fwdvol_t": res.get("partial_ic_conf_fwdvol_t"),
                "partial_ic_conf_fwdmdd": res.get("partial_ic_conf_fwdmdd"),
                "holds": bool(res.get("holds")),
            })
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[risk-signal] 子池稳健性检验失败: {e}")
            out["trials"].append({"available": False, "reason": str(e)})

    out["n_subsets_usable"] = int(usable)
    out["n_holds"] = int(holds)
    out["hold_rate"] = _round(holds / usable) if usable else None
    out["available"] = bool(usable > 0)
    if not out["available"]:
        out["reason"] = "所有子池均不可用（样本不足）"
        return out
    rate = out["hold_rate"] or 0.0
    if usable >= 5 and rate >= 0.8:
        out["robustness_verdict"] = "robust"
    elif usable >= 5 and rate >= 0.5:
        out["robustness_verdict"] = "suggestive"
    else:
        out["robustness_verdict"] = "fragile"
    out["verdict_note"] = (
        "'robust' 才够格作为立项依据（仍须人工签字）；'suggestive' 只能作"
        "下一步实验方向；'fragile' 不得外推。"
    )
    return out


def _risk_core(d: pd.DataFrame, horizon_days: int) -> Dict[str, Any]:
    """子池内层：单池增量读数 + holds 判定（阈值与主口径一致）。"""
    p = d["_p"].to_numpy(dtype=float)
    conf = np.abs(p - 0.5) * 2.0
    fwd_vol = d["_fwd_vol"].to_numpy(dtype=float)
    fwd_mdd = d["_fwd_mdd"].to_numpy(dtype=float)
    trail = d["_trail_vol"].to_numpy(dtype=float)
    n = int(len(d))
    n_eff = _effective_n(n, int(horizon_days))
    pic_vol = _partial_spearman(conf, fwd_vol, trail)
    pic_mdd = _partial_spearman(conf, fwd_mdd, trail)
    t_vol = _t_stat(pic_vol, n_eff)
    t_mdd = _t_stat(pic_mdd, n_eff)
    holds = bool(_judge_increment(pic_vol, n_eff, positive=True)
                 or _judge_increment(pic_mdd, n_eff, positive=False))
    return {
        "n_samples": n,
        "n_effective": n_eff,
        "partial_ic_conf_fwdvol": _round(pic_vol),
        "partial_ic_conf_fwdvol_t": _round(t_vol, 4),
        "partial_ic_conf_fwdmdd": _round(pic_mdd),
        "partial_ic_conf_fwdmdd_t": _round(t_mdd, 4),
        "holds": holds,
    }


# ----------------------------------------------------------------------
# 汇总报告
# ----------------------------------------------------------------------
def build_report(data: Dict[str, pd.DataFrame], probabilities: Dict[int, pd.Series],
                 config: Dict[str, Any], horizons: Sequence[int],
                 stability: bool = True, n_subsets: int = 12,
                 subset_size: int = 18, seed: int = 0) -> Dict[str, Any]:
    """逐周期增量读数 + 子池稳健性 → 总结论。"""
    per_horizon: Dict[str, Any] = {}
    for h in horizons:
        proba = probabilities.get(int(h))
        ds = build_risk_dataset(data, int(h))
        if proba is None or ds is None or ds.empty:
            per_horizon[str(h)] = {"available": False, "reason": "缺概率或风险数据"}
            continue
        proba = proba.copy()
        proba["date"] = pd.to_datetime(proba["date"]).dt.normalize()
        ds["date"] = pd.to_datetime(ds["date"]).dt.normalize()
        merged = ds.merge(proba, on=["date", "_symbol"], how="inner")
        if merged.empty:
            per_horizon[str(h)] = {"available": False, "reason": "合并后为空"}
            continue
        # 归一索引后交给 risk_informativeness
        merged = merged.reset_index(drop=True)
        probs = pd.Series(merged["_p"].to_numpy(dtype=float))
        res = risk_informativeness(
            merged.drop(columns=["_p"]), probs, int(h))
        per_horizon[str(h)] = res

    avail = [v for v in per_horizon.values() if v.get("available")]
    hits = [v for v in avail
            if v.get("has_vol_increment") or v.get("has_drawdown_increment")]
    report: Dict[str, Any] = {
        "kind": "risk_signal_informativeness",
        "affects_gate": False,
        "readonly": True,
        "boundary": "只报数：不改门禁 / 权重 / 池 / 配置；本模块只回答'值不值得立项'。",
        "criterion": ("增量判据 = 控制 trailing-vol 基线后的偏秩相关（partial IC）；"
                      "方向：conf→未来波动 需 > 0、conf→未来回撤 需 < 0；"
                      "且须**效应量** |IC| ≥ 0.10（防大 n 抬小相关）"
                      "**且** |t| ≥ 2（t 已按重叠周期折算有效 n）。"),
        "effect_size_floor": MIN_EFFECT_SIZE,
        "min_t": MIN_T,
        "n_symbols": int(len(data or {})),
        "per_horizon": per_horizon,
        "n_horizons_available": len(avail),
        "n_horizons_with_increment": len(hits),
    }

    stability_block: Dict[str, Any]
    if stability:
        # 稳健性沿用 mid 周期（10d）—— 与 regime-signal 同口径
        h_mid = int(sorted(horizons)[len(horizons) // 2]) if horizons else 10
        stability_block = risk_stability_check(
            data, config, horizon_days=h_mid, n_subsets=int(n_subsets),
            subset_size=int(subset_size), seed=int(seed))
    else:
        stability_block = {"available": False, "reason": "未做稳健性检验"}

    report["stability"] = stability_block

    if not avail:
        report["conclusion"] = "unavailable"
        report["recommendation"] = "读数不可用，无法判断风险预警是否值得立项。"
    elif not hits:
        report["conclusion"] = "no_risk_increment"
        report["recommendation"] = (
            "模型输出**不提供超出朴素 trailing-vol 的风险增量** ⇒ "
            "若要做波动/回撤预警，用朴素基线即可，**不必**为本模型的输出立项。"
        )
    else:
        rob = stability_block.get("robustness_verdict")
        if rob == "robust":
            report["conclusion"] = "risk_increment_robust"
            report["recommendation"] = (
                "增量风险信号在子池上稳健 ⇒ 够格作为**立项依据**（仍须人工签字）；"
                "下一步可做风险预警的产品化评估（仍只读、不接门禁）。"
            )
        elif rob == "suggestive":
            report["conclusion"] = "risk_increment_suggestive"
            report["recommendation"] = (
                "增量风险信号方向成立但子池稳健性只到 suggestive ⇒ "
                "够格作**下一步实验方向**，不够格直接立项/改口径。"
            )
        else:
            report["conclusion"] = "risk_increment_fragile"
            report["recommendation"] = (
                "增量风险信号在子池上不稳健（fragile）⇒ 不得外推，"
                "不支持据此立项。"
            )
    return report
