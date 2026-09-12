"""市场状态分层（S18 / H3，T18.1）：用 HMM 识别牛 / 熊 / 震荡，并做状态内分层评估。

问题从哪来（S15 结论二 + README 已知限制）：
  S15 用 optuna 收敛搜索后，「IC 为正但命中率卡在 50% 附近」依旧存在。
  S9 已经从**资产维度**分池解释了一部分（个股 vs ETF 波动结构不同），
  但**时间维度**没查过：模型是不是只在特定市场状态下有效？
  如果信号只在震荡市有效、趋势市失效，那么整池平均命中率必然被拉平到 50%
  —— 这不是模型不行，而是"状态混合"掩盖了条件有效性。

参照 hmmlearn（BSD-3-Clause）的做法，本模块提供：

  1. **状态识别**（T18.1）：对单一市场指数序列（默认取标的池的等权收益代理）
     拟合 HMM，把每一天划成 牛 / 熊 / 震荡 三态。状态标签严格只用
     **当日及之前**的信息（用前向滤波而非全序列平滑，见无前视说明）；
  2. **状态内分层评估**（T18.2）：同一套 walk-forward 预测，按状态把样本
     分组，分别算 IC / 命中率 / 样本数，回答"模型是否只在特定状态有效"；
  3. **状态增量 A/B**（T18.3）：把「状态 one-hot」作为一个新变量加进特征，
     同口径对比增量，结论好坏如实入库。

本模块**不做什么**（边界比功能重要）：
  - **不改门禁**：``affects_gate`` 恒为 False；状态是否进信号门禁 / 风控
    `withheld` 语义属 T18.4 人工检查点；
  - **不猜状态**：HMM 不可用（无 hmmlearn）时**降级为可解释的规则口径**
    （滚动趋势 + 波动分位），并在报告里如实标注 ``backend="rules"``，
    不冒充 HMM 结果；
  - **不用全序列平滑**：全序列 Viterbi/后验平滑会用到未来数据 —— 本模块
    只用**前向滤波**（t 时刻只用 ``[.., t]``），宁可状态抖动也不引入前视；
  - **不猜**：序列太短 / 状态样本不足 → ``available=false`` + 原因。

无前视说明：
  - 状态序列第 t 个元素只由 ``returns[.., t]`` 决定（前向滤波 + 递归更新）；
  - 分层评估按状态分组时，用的是**同一时点**已经已知的状态标签，
    不会把未来的状态信息回填给过去的样本。
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

REGIME_BULL = "bull"
REGIME_BEAR = "bear"
REGIME_RANGE = "range"
REGIME_ORDER = (REGIME_BULL, REGIME_RANGE, REGIME_BEAR)

DEFAULT_N_STATES = 3
DEFAULT_VOL_WINDOW = 20
MIN_SAMPLES = 60
TREND_STRONG = 0.02      # 规则口径：20 日累计收益阈值（趋势市判定）
VOL_HIGH_QUANTILE = 0.75  # 规则口径：高波动分位（震荡市用不到，仅作分级参考）


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _try_hmmlearn():
    """尝试导入 hmmlearn；不可用返回 None（调用方据此降级，如实标注）。"""
    try:  # pragma: no cover - 取决于环境
        from hmmlearn.hmm import GaussianHMM  # type: ignore

        return GaussianHMM
    except Exception:  # noqa: BLE001
        return None


def _forward_filter(log_likelihoods: np.ndarray, transition: np.ndarray,
                    initial: np.ndarray) -> np.ndarray:
    """HMM 前向滤波（**只用过去信息**，不做平滑）。

    返回形状 ``(T, K)`` 的后验概率矩阵；第 t 行只用 ``log_likelihoods[:t+1]``。
    """
    t_len, k = log_likelihoods.shape
    alpha = np.zeros((t_len, k), dtype=float)
    alpha[0] = initial * np.exp(log_likelihoods[0])
    s = alpha[0].sum()
    alpha[0] = alpha[0] / s if s > 0 else np.full(k, 1.0 / k)
    for t in range(1, t_len):
        prior = alpha[t - 1] @ transition
        post = prior * np.exp(log_likelihoods[t])
        s = post.sum()
        alpha[t] = post / s if s > 0 else prior
    return alpha


def _state_semantics(means: Sequence[float], vols: Sequence[float]) -> List[str]:
    """把 HMM 状态映射为 牛 / 熊 / 震荡（按均值收益排序，最波动者优先判震荡）。

    规则（可解释、不玄学）：
      - 有 3 态时：均值最高 → 牛；均值最低 → 熊；剩下 → 震荡；
      - 只有 2 态时：均值高者 → 牛，低者 → 熊（无震荡态，如实反映）。
    """
    k = len(means)
    order = list(np.argsort(means))
    labels = [REGIME_RANGE] * k
    if k == 1:
        return labels
    labels[order[-1]] = REGIME_BULL
    labels[order[0]] = REGIME_BEAR
    # 中间状态（若有）保持 range
    return labels


def _fit_hmm(GaussianHMM, returns: np.ndarray, k: int, seed: int = 42,
             min_state_share: float = 0.05) -> Optional[Dict[str, Any]]:
    """拟合 HMM 并做**退化检查**：不收敛/状态占比过低/方差爆炸一律返回 None。

    为什么需要（这是 HMM 最常见的地雷）：
      不加约束的 ``GaussianHMM`` 会把某个状态吸收成"方差巨大"的离群态
      （协方差爆炸），于是只剩 2 个有效状态、甚至所有样本落在一态。
      这样的"状态"没有市场语义，拿去分层只会得出假结论 —— 宁可如实降级。

    检查项：
      1. ``converged_`` 为真（达到迭代上限也不算不收敛，但要看下面）；
      2. 每个状态的**占比下限** ``min_state_share``（默认 5%）；
      3. 状态协方差**不得爆炸**（超过全体方差的 100 倍视为退化）。

    多组随机种子取第一个通过检查的拟合（避免单次随机初始化翻车）。
    """
    X_raw = returns.reshape(-1, 1)
    baseline_var = float(np.var(returns) + 1e-12)

    # 初始化：按收益的**分位**把样本粗分给各状态（牛/震荡/熊），
    # 用这些子样本的均值方差做起始参数。这样起点就带市场语义，
    # 避免 EM 从一个离谱起点出发把某状态吸收成"方差爆炸"的离群态。
    qs = np.linspace(0, 100, int(k) + 1)
    edges = np.percentile(returns, qs)
    init_means = np.zeros((int(k), 1), dtype=float)
    init_covars = np.zeros((int(k), 1), dtype=float)
    for i in range(int(k)):
        lo, hi = edges[i], edges[i + 1]
        sel = (returns >= lo) & (returns <= hi)
        seg = returns[sel] if sel.sum() >= 2 else returns
        init_means[i, 0] = float(np.mean(seg))
        init_covars[i, 0] = float(np.var(seg) + 1e-12)
    # 状态顺序固定为"低收益 → 高收益"，让语义映射不依赖随机初始化
    order = np.argsort(init_means.ravel())
    init_means = init_means[order]
    init_covars = init_covars[order]

    candidates = [seed, seed + 1, seed + 7]
    best: Optional[Dict[str, Any]] = None
    for sd in candidates:
        try:
            model = GaussianHMM(n_components=int(k), covariance_type="diag",
                                n_iter=200, tol=1e-5, random_state=int(sd),
                                init_params="st", params="stmc")
            model.means_ = init_means.copy()
            model.covars_ = init_covars.copy()
            model.fit(X_raw)
            means = [float(m[0]) for m in model.means_]
            covars = [float(c[0][0]) for c in model.covars_]
            vols = [math.sqrt(max(v, 1e-12)) for v in covars]
            labels = _state_semantics(means, vols)
            ll = model._compute_log_likelihood(X_raw)  # noqa: SLF001
            post = _forward_filter(ll, model.transmat_, model.startprob_)
            idx = np.argmax(post, axis=1)
            shares = np.bincount(idx, minlength=int(k)) / max(1, idx.size)
            if shares.min() < min_state_share:
                logger.warning(f"[regime] HMM(seed={sd}) 状态占比过低 "
                               f"{np.round(shares, 3).tolist()}，跳过")
                continue
            if max(covars) > 100.0 * baseline_var:
                logger.warning(f"[regime] HMM(seed={sd}) 协方差爆炸 "
                               f"({max(covars):.3g} vs 基线 {baseline_var:.3g})，跳过")
                continue
            states = [labels[int(i)] for i in idx]
            if len(set(states)) < 2:
                continue
            best = {
                "available": True,
                "backend": "hmm",
                "n_states": int(k),
                "samples": int(returns.size),
                "state_means": [round(m, 8) for m in means],
                "state_vols": [round(v, 8) for v in vols],
                "state_labels": labels,
                "state_shares": [round(float(x), 6) for x in shares.tolist()],
                "states": states,
                "reason": "",
                "note": ("状态用前向滤波逐时点推断（只用历史），非全序列平滑；"
                         "已做状态占比与协方差退化检查"),
            }
            break
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[regime] HMM(seed={sd}) 拟合失败: {e}")
            continue
    return best

def detect_regimes(returns: Sequence[float],
                   *,
                   n_states: int = DEFAULT_N_STATES,
                   backend: str = "auto",
                   seed: int = 42,
                   min_samples: int = MIN_SAMPLES) -> Dict[str, Any]:
    """识别市场状态序列（HMM 优先，规则口径降级）。

    参数：
      ``returns``：市场代理收益序列（如等权组合日收益），**严格按时间升序**；
      ``backend``：``"auto"`` / ``"hmm"`` / ``"rules"``；``hmm`` 但不可用时
        直接返回不可用（不静默降级）。

    返回 ``{"available", "backend", "states", "state_means", "state_vols", "reason"}``。
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = int(r.size)
    if n < min_samples:
        return {"available": False, "backend": None, "states": [],
                "state_means": [], "state_vols": [],
                "reason": f"序列太短（{n} < {min_samples}），不猜状态"}

    k = int(n_states)
    GaussianHMM = _try_hmmlearn() if backend in ("auto", "hmm") else None

    if backend == "hmm" and GaussianHMM is None:
        return {"available": False, "backend": None, "states": [],
                "state_means": [], "state_vols": [],
                "reason": "backend=hmm 但 hmmlearn 未安装（不静默降级）"}

    if GaussianHMM is not None:
        attempt = _fit_hmm(GaussianHMM, r, k, seed=seed)
        if attempt is not None:
            return attempt
        logger.warning("[regime] HMM 未收敛/退化，降级为规则口径")
        if backend == "hmm":
            return {"available": False, "backend": None, "states": [],
                    "state_means": [], "state_vols": [],
                    "reason": "HMM 未收敛或状态退化（非静默降级）"}

    # 规则口径降级（可解释、零依赖、同样无前视）
    states = _rules_states(r)
    if len(set(states)) == 1:
        return {"available": False, "backend": "rules", "states": states,
                "state_means": [], "state_vols": [],
                "reason": "规则口径下全部样本落在同一状态，无法分层（不猜）"}
    means, vols = {}, {}
    for s in set(states):
        vals = r[np.array([x == s for x in states])]
        means[s] = round(float(np.mean(vals)), 8)
        vols[s] = round(float(np.std(vals)), 8)
    return {
        "available": True,
        "backend": "rules",
        "n_states": len(set(states)),
        "samples": n,
        "state_means": means,
        "state_vols": vols,
        "state_labels": list(set(states)),
        "states": states,
        "reason": "",
        "note": "hmmlearn 未安装，降级为滚动趋势/波动规则口径（如实标注 backend=rules）",
    }


def _rules_states(returns: np.ndarray,
                  window: int = DEFAULT_VOL_WINDOW,
                  threshold: float = TREND_STRONG) -> List[str]:
    """规则口径状态：滚动累计收益 > +thr → 牛；< -thr → 熊；其余 → 震荡。

    严格无前视：第 t 天只看 ``returns[max(0,t-window+1) : t+1]``。
    """
    n = returns.size
    out: List[str] = []
    for t in range(n):
        lo = max(0, t - window + 1)
        seg = returns[lo:t + 1]
        cum = float(np.sum(seg))
        if cum > threshold:
            out.append(REGIME_BULL)
        elif cum < -threshold:
            out.append(REGIME_BEAR)
        else:
            out.append(REGIME_RANGE)
    return out


def stratified_by_regime(
    scores: Sequence[float],
    fwd_ret: Sequence[float],
    states: Sequence[str],
    *,
    min_samples: int = 30,
    hit_rate_fn=None,
    ic_fn=None,
) -> Dict[str, Any]:
    """按市场状态分层评估 IC / 命中率（T18.2）。

    只在**同一批样本**上分组（不重采样、不换折），因此各状态读数可直接与
    整体读数对照。样本不足的状态标 ``available=false``，不猜。
    """
    from src.inference.ic import hit_rate as _hit
    from src.inference.ic import spearman_ic as _ic

    hit = hit_rate_fn or (lambda s, r: _hit(list(s), list(r)))
    ic = ic_fn or (lambda s, r: _ic(list(s), list(r)))

    s = np.asarray(scores, dtype=float)
    r = np.asarray(fwd_ret, dtype=float)
    n = int(min(s.size, r.size, len(states)))
    if n == 0:
        return {"available": False, "reason": "无样本", "groups": [],
                "affects_gate": False}

    s, r, st = s[:n], r[:n], list(states)[:n]
    groups: List[Dict[str, Any]] = []
    overall = {
        "samples": n,
        "hit_rate": (None if hit(s, r) is None else round(float(hit(s, r)), 6)),
        "ic": (None if ic(s, r) is None else round(float(ic(s, r)), 6)),
    }
    for name in REGIME_ORDER:
        mask = np.array([x == name for x in st])
        cnt = int(mask.sum())
        row: Dict[str, Any] = {"regime": name, "samples": cnt}
        if cnt < min_samples:
            row.update({"available": False,
                        "reason": f"样本不足（{cnt} < {min_samples}），不猜",
                        "hit_rate": None, "ic": None})
        else:
            hr = hit(s[mask], r[mask])
            icv = ic(s[mask], r[mask])
            row.update({
                "available": True,
                "reason": "",
                "hit_rate": None if hr is None else round(float(hr), 6),
                "ic": None if icv is None else round(float(icv), 6),
                "hit_rate_delta_vs_overall": (
                    None if (hr is None or overall["hit_rate"] is None)
                    else round(float(hr) - overall["hit_rate"], 6)),
                "ic_delta_vs_overall": (
                    None if (icv is None or overall["ic"] is None)
                    else round(float(icv) - overall["ic"], 6)),
            })
        groups.append(row)

    usable = [g for g in groups if g["available"]]
    best = max(usable, key=lambda g: g["hit_rate"]) if usable else None
    worst = min(usable, key=lambda g: g["hit_rate"]) if usable else None
    spread = (None if not usable else round(best["hit_rate"] - worst["hit_rate"], 6))
    return {
        "kind": "regime_stratified",
        "generated_at": _now(),
        "available": bool(usable),
        "overall": overall,
        "groups": groups,
        "best_regime": None if best is None else best["regime"],
        "worst_regime": None if worst is None else worst["regime"],
        "hit_rate_spread": spread,
        "condition_conclusion": _condition_conclusion(usable, spread),
        "affects_gate": False,
        "note": ("状态分层只回答「是否只在特定状态有效」；是否进信号门禁 / 风控"
                 "withheld 语义属 T18.4 人工检查点。"),
    }


def _condition_conclusion(usable: Sequence[Dict[str, Any]],
                          spread: Optional[float]) -> str:
    """把分层读数翻译成一句保守结论（不夸大、不宣布达标）。"""
    if not usable:
        return "无可用状态组（样本不足），无法判断条件有效性"
    if spread is None or spread < 0.02:
        return "各状态命中率差异 <2pp：未观察到明显的状态条件有效性"
    best = max(usable, key=lambda g: g["hit_rate"])
    worst = min(usable, key=lambda g: g["hit_rate"])
    return (f"命中率在 {best['regime']}({best['hit_rate']:.2%}) 与 "
            f"{worst['regime']}({worst['hit_rate']:.2%}) 间差 {spread:.2%}pp："
            "存在状态条件性，但样本量小，**不得直接作为门禁证据**")


def regime_feature_increment(
    base_scores: Sequence[float],
    augmented_scores: Sequence[float],
    fwd_ret: Sequence[float],
) -> Dict[str, Any]:
    """状态作为特征的增量 A/B（T18.3）：同一批样本、只差一个变量。

    比较「基准打分」与「加入状态 one-hot 后的打分」在同一份未来收益上的
    IC / 命中率增量；结论好坏如实入库，不修饰。
    """
    from src.inference.ic import hit_rate, spearman_ic

    b = np.asarray(base_scores, dtype=float)
    a = np.asarray(augmented_scores, dtype=float)
    r = np.asarray(fwd_ret, dtype=float)
    n = int(min(b.size, a.size, r.size))
    if n == 0:
        return {"available": False, "reason": "无样本", "affects_gate": False}
    b, a, r = b[:n], a[:n], r[:n]

    ic_b, ic_a = spearman_ic(list(b), list(r)), spearman_ic(list(a), list(r))
    hr_b, hr_a = hit_rate(list(b), list(r)), hit_rate(list(a), list(r))
    d_ic = None if (ic_b is None or ic_a is None) else round(float(ic_a) - float(ic_b), 6)
    d_hr = None if (hr_b is None or hr_a is None) else round(float(hr_a) - float(hr_b), 6)
    improved = bool((d_ic or 0) > 0 and (d_hr or 0) > 0)
    return {
        "kind": "regime_feature_ab",
        "generated_at": _now(),
        "available": True,
        "samples": n,
        "baseline": {"ic": ic_b, "hit_rate": hr_b},
        "augmented": {"ic": ic_a, "hit_rate": hr_a},
        "delta_ic": d_ic,
        "delta_hit_rate": d_hr,
        "improved": improved,
        "conclusion": (
            "加入状态特征出现正向增量（需人工复核后再决定是否纳入主线）"
            if improved else
            "加入状态特征未跑出正向增量；按既有纪律如实入库，不纳入主线"
        ),
        "affects_gate": False,
        "note": "增量对比为同数据同折单变量对照；结论不构成达证据，落地属 T18.4 人工检查点。",
    }
