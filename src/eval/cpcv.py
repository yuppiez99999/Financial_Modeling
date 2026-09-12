"""组合式净化交叉验证 CPCV 与过拟合概率（S17 / H2，T17.1）。

问题从哪来（S11 / S12 / S13 / S15 的共同墙）：
  所有 A/B 都用 walk-forward 折。折数少（3~5），每个读数都建立在**同一份
  历史**上；再叠加"反复试口径"，就会出现「某个组合看着达标」而实际是
  选择了噪声。S13 只把"试了多少次"登记下来，S15 也明说"读数未经多重比较
  校正不得引用为达标证据" —— 但**选择自由度本身没有被量化成过拟合概率**。

参照 López de Prado《Advances in Financial Machine Learning》（CPCV / DSR /
PBO 一节）的做法，本模块提供：

  1. **CPCV 折生成**：把时间轴切成 ``N`` 组，任取 ``k`` 组作测试、其余作训练，
     得到 ``C(N, k)`` 条路径 —— 比单条 walk-forward 覆盖更多"训练/测试"
     组合，且每条路径都保持**时序方向**（训练段可以有多个不连续块，但
     训练与测试之间一定有净化间隔）；
  2. **purge + embargo**：训练样本与测试样本的**标签窗口**若重叠必须剔除
     （purge）；测试段之后紧邻的 ``embargo`` 个样本也从训练集中剔除
     （防串联相关），对齐 ``00_kickoff/leakage_checklist.md``；
  3. **过拟合概率**：给出 DSR 风格的**选择偏差校正指标**（对同一组候选
     策略的多次评估做收缩），以及 PBO（回测过拟合概率：训练段最优策略
     在测试段跑输中位数的比例）。

本模块**不做什么**（边界比功能重要）：
  - **不改门禁**：``affects_gate`` 恒为 False，不写配置、不改判据；
  - **不做显著性声明**：CPCV 只产出**参考读数**，是否引入过拟合概率下限
    属 T17.4 人工检查点；
  - **不猜**：路径数不足 / 候选不足 → ``available=false`` + 原因；
  - **不实现完整 DSR 论文公式**：本模块给出的是**可复算的近似收缩指标**，
    报告里如实标注 ``method="cpcv_shrinkage"``，不冒充 Sharpe 的 DSR 精确解。

无前视说明：
  折生成只依赖样本**索引**；purge/embargo 只做**剔除**（不新增样本）。
"""
from __future__ import annotations

import itertools
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_N_BLOCKS = 6
DEFAULT_K_TEST = 2
DEFAULT_EMBARGO = 5
MIN_PATHS = 3


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cpcv_paths(n_samples: int, n_blocks: int = DEFAULT_N_BLOCKS,
               k_test: int = DEFAULT_K_TEST,
               horizon_days: int = 1,
               embargo: int = DEFAULT_EMBARGO) -> List[Dict[str, Any]]:
    """生成 CPCV 路径（组合式：任取 ``k_test`` 组作测试）。

    返回每条路径的:
      - ``train_idx`` / ``test_idx``：样本索引（已 purge + embargo）；
      - ``purged``：因标签窗口重叠被剔除的样本数；
      - ``embargoed``：因 embargo 被剔除的样本数。

    参数不足（``n_blocks < k_test``、样本太少、路径数 < ``MIN_PATHS``）时返回
    空列表 —— 由调用方标不可用，不猜。
    """
    n = int(n_samples)
    nb = int(n_blocks)
    k = int(k_test)
    if n <= 0 or nb <= 0 or k <= 0 or k >= nb:
        return []
    if n < nb * 20:
        return []
    bounds = [int(round(i * n / nb)) for i in range(nb + 1)]
    bounds[-1] = n
    if any(bounds[i + 1] <= bounds[i] for i in range(nb)):
        return []

    h = max(0, int(horizon_days) - 1)   # 标签窗口长度（天）
    emb = max(0, int(embargo))
    all_idx = np.arange(n)
    paths: List[Dict[str, Any]] = []
    for combo in itertools.combinations(range(nb), k):
        test_parts = [np.arange(bounds[b], bounds[b + 1]) for b in combo]
        test_idx = np.concatenate(test_parts) if test_parts else np.array([], dtype=int)

        # 净化的训练集：先取全体，再剔除标签窗口重叠与 embargo 区
        drop = np.zeros(n, dtype=bool)
        for b in combo:
            lo, hi = bounds[b], bounds[b + 1]
            # purged：测试段的标签窗口 [lo-h, hi+h) 与训练样本重叠一律剔除
            drop[max(0, lo - h):min(n, hi + h)] = True
            # embargo：测试段之后紧邻的 emb 个训练样本剔除
            drop[hi:min(n, hi + emb)] = True
        train_mask = ~drop
        train_idx = all_idx[train_mask]
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        purged = int(np.sum(drop & ~np.isin(all_idx, test_idx)))
        paths.append({
            "test_intervals": [[bounds[b], bounds[b + 1]] for b in combo],
            "test_blocks": [int(b) for b in combo],
            "train_blocks": [int(b) for b in range(nb) if b not in combo],
            "train_idx": train_idx,
            "test_idx": np.sort(test_idx),
            "purged": purged,
            "embargoed": int(sum(min(emb, n - bounds[b + 1]) for b in combo)),
        })
    if len(paths) < MIN_PATHS:
        return []
    return paths


def cpcv_splits(n_samples: int, n_blocks: int = DEFAULT_N_BLOCKS,
                k_test: int = DEFAULT_K_TEST, horizon_days: int = 1,
                embargo: int = DEFAULT_EMBARGO) -> List[Tuple[np.ndarray, np.ndarray]]:
    """CPCV 路径 → ``(train_idx, test_idx)`` 列表（接口兼容 walk_forward_splits）。"""
    return [(p["train_idx"], p["test_idx"]) for p in cpcv_paths(
        n_samples, n_blocks=n_blocks, k_test=k_test,
        horizon_days=horizon_days, embargo=embargo)]


def purge_embargo_audit(n_samples: int, train_idx: Sequence[int],
                        test_idx: Sequence[int], horizon_days: int,
                        embargo: int = DEFAULT_EMBARGO,
                        test_blocks: Optional[Sequence[Tuple[int, int]]] = None) -> Dict[str, Any]:
    """审计一条路径的净化结果：**训练样本的标签窗口不得与测试段重叠**。

    关键：CPCV 的测试段是**多个不连续区间**（如第 0、3 组），中间的训练样本
    并不构成泄漏 —— 必须**逐测试区间**判定，不能用 ``[min, max]`` 一把量，
    否则会把合法训练样本误判成重叠（这是本审计最容易写错的地方）。

    ``test_blocks`` 为 ``[(lo, hi), ...]`` 的测试区间；缺省时退化为把
    ``test_idx`` 当作一个连续区间（单块口径）。

    返回 ``{"ok", "overlaps", "reason", "per_interval"}``。
    """
    n = int(n_samples)
    tr = np.asarray(train_idx, dtype=int)
    te = np.asarray(test_idx, dtype=int)
    if tr.size == 0 or te.size == 0:
        return {"ok": False, "overlaps": 0, "reason": "训练或测试为空",
                "per_interval": []}

    h = max(0, int(horizon_days) - 1)
    emb = max(0, int(embargo))

    intervals: List[Tuple[int, int, int]] = []   # (lo, hi, purge 下界)
    if test_blocks:
        for lo, hi in test_blocks:
            intervals.append((int(lo), int(hi), max(0, int(lo) - h)))
    else:
        intervals.append((int(te.min()), int(te.max()) + 1, max(0, int(te.min()) - h)))

    per_interval: List[Dict[str, Any]] = []
    total_overlap = 0
    for (lo, hi, keep_end) in intervals:
        # 训练样本 t 的标签窗口 [t, t+h] 若与测试区间 [lo, hi) 相交 → 泄漏；
        # 测试段之后 emb 天内的训练样本也剔除（embargo）。
        # 判定：t 必须满足 t + h < lo 或 t >= hi + emb
        bad = (tr + h >= lo) & (tr < hi + emb)
        n_bad = int(np.sum(bad))
        total_overlap += n_bad
        per_interval.append({"test_interval": [lo, hi], "overlaps": n_bad})

    return {
        "ok": total_overlap == 0,
        "overlaps": total_overlap,
        "reason": "" if total_overlap == 0
                  else f"{total_overlap} 个训练样本标签窗口/embargo 重叠",
        "per_interval": per_interval,
    }


def deflated_shrinkage(scores: Sequence[float], n_trials: int,
                       return_std: Optional[float] = None) -> Dict[str, Any]:
    """选择偏差收缩指标（DSR 风格的**近似**，方法名如实标注）。

    - 输入：**一组候选**在同一评估口径下的得分（如各超参/各口径的 IC）；
    - 收缩公式：``shrunk = best − E[max of n_trials noise]``，其中噪声项用
      ``sqrt(2·ln(n_trials)) · std``（极值理论的一阶近似）；
    - 输出：原始最优、收缩后最优、以及是否仍为正（正 = 未被选择偏差吃掉）。

    这只是**一阶近似**，不替代 Sharpe 比率的 DSR 精确解（见模块 docstring 边界）。
    """
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return {"available": False, "method": "cpcv_shrinkage", "reason": "无有效得分"}
    nt = max(1, int(n_trials))
    std = float(return_std) if return_std is not None else (
        float(np.std(s, ddof=1)) if s.size > 1 else 0.0)
    penalty = float(np.sqrt(2.0 * np.log(max(nt, 2))) * std)
    best = float(np.max(s))
    return {
        "available": True,
        "method": "cpcv_shrinkage",
        "n_candidates": int(s.size),
        "n_trials": nt,
        "best_raw": round(best, 6),
        "penalty": round(penalty, 6),
        "best_shrunk": round(best - penalty, 6),
        "survives": bool(best - penalty > 0),
        "reason": "",
    }


def pbo(candidate_scores: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """回测过拟合概率 PBO：训练段最优策略在测试段跑输中位数的比例。

    Args:
        candidate_scores: ``[[策略1 各路径训练得分], [策略1 各路径测试得分], ...]``
            形如 ``[(train_scores, test_scores), ...]`` 每个策略一列数组。

    实现（CSCV 简化版）：对每条路径，取训练段最优策略，看它在测试段是否
    低于该路径测试段的中位数；PBO = 这样的路径占比。0 表示无过拟合迹象，
    0.5 表示和随机选策略无异。
    """
    pairs = [p for p in candidate_scores if p and len(p) == 2]
    if not pairs:
        return {"available": False, "pbo": None, "reason": "无候选-路径矩阵"}
    try:
        train_mat = np.vstack([np.asarray(p[0], dtype=float) for p in pairs])
        test_mat = np.vstack([np.asarray(p[1], dtype=float) for p in pairs])
    except Exception:  # noqa: BLE001
        return {"available": False, "pbo": None, "reason": "候选-路径矩阵形状不一致"}
    n_paths = min(train_mat.shape[1], test_mat.shape[1])
    n_cand = min(train_mat.shape[0], test_mat.shape[0])
    if n_paths < 2 or n_cand < 2:
        return {"available": False, "pbo": None,
                "reason": f"路径数 {n_paths} 或候选数 {n_cand} 不足（需 ≥2）"}
    train_mat, test_mat = train_mat[:n_cand, :n_paths], test_mat[:n_cand, :n_paths]
    bad = 0
    valid = 0
    details: List[Dict[str, Any]] = []
    for j in range(n_paths):
        tr, te = train_mat[:, j], test_mat[:, j]
        if not np.isfinite(tr).any() or not np.isfinite(te).any():
            continue
        best = int(np.nanargmax(tr))
        med = float(np.nanmedian(te))
        under = bool(te[best] < med)
        valid += 1
        bad += int(under)
        details.append({"path": j, "best_candidate": best,
                        "test_score": round(float(te[best]), 6),
                        "test_median": round(med, 6), "underperforms": under})
    if valid == 0:
        return {"available": False, "pbo": None, "reason": "无有效路径"}
    value = bad / valid
    return {
        "available": True,
        "pbo": round(value, 6),
        "n_paths": valid,
        "n_candidates": n_cand,
        "interpretation": (
            "0 = 未见过拟合迹象；0.5 ≈ 与随机选策略无异" if value < 0.5
            else "≥0.5 = 训练段最优在测试段大概率跑输中位数，过拟合迹象明显"
        ),
        "paths": details,
        "reason": "",
    }


def cpcv_evaluate(
    scores_per_path: Sequence[float],
    *,
    n_samples: int,
    n_blocks: int = DEFAULT_N_BLOCKS,
    k_test: int = DEFAULT_K_TEST,
    horizon_days: int = 1,
    embargo: int = DEFAULT_EMBARGO,
    n_trials: int = 1,
    candidate_scores: Optional[Sequence[Sequence[float]]] = None,
) -> Dict[str, Any]:
    """CPCV 评估报告：路径净化审计 + 得分分布 + 收缩指标 + PBO。

    ``scores_per_path``：**候选策略**在每条路径测试段上的得分（IC 等）。
    """
    paths = cpcv_paths(n_samples, n_blocks=n_blocks, k_test=k_test,
                       horizon_days=horizon_days, embargo=embargo)
    if not paths:
        return {
            "kind": "cpcv_evaluation", "generated_at": _now(),
            "available": False,
            "reason": (f"路径不足：样本 {n_samples} / 组数 {n_blocks} / 取 k={k_test} "
                       f"/ horizon={horizon_days}，需 ≥{MIN_PATHS} 条路径（不猜）"),
            "affects_gate": False, "paths": [], "scores": [],
        }

    s = np.asarray(scores_per_path, dtype=float)
    s = s[np.isfinite(s)]
    audits = []
    for p in paths:
        audits.append(purge_embargo_audit(
            n_samples, p["train_idx"], p["test_idx"], horizon_days, embargo=embargo,
            test_blocks=[tuple(x) for x in p.get("test_intervals", [])]))
    audit_ok = bool(audits) and all(a["ok"] for a in audits)

    shrink = deflated_shrinkage(s.tolist(), n_trials=n_trials) if s.size else {
        "available": False, "method": "cpcv_shrinkage", "reason": "无路径得分"}
    pbo_res = pbo(candidate_scores) if candidate_scores else {
        "available": False, "pbo": None, "reason": "未提供候选-路径矩阵"}

    return {
        "kind": "cpcv_evaluation",
        "generated_at": _now(),
        "available": True,
        "n_samples": int(n_samples),
        "n_blocks": int(n_blocks),
        "k_test": int(k_test),
        "horizon_days": int(horizon_days),
        "embargo": int(embargo),
        "n_paths": len(paths),
        "paths": [{"test_blocks": p["test_blocks"], "train_blocks": p["train_blocks"],
                   "test_intervals": p.get("test_intervals"),
                   "n_train": int(p["train_idx"].size), "n_test": int(p["test_idx"].size),
                   "purged": p["purged"], "embargoed": p["embargoed"]} for p in paths],
        "purge_audit": {"ok": audit_ok, "details": audits},
        "scores": [round(float(x), 6) for x in s],
        "score_mean": None if s.size == 0 else round(float(np.mean(s)), 6),
        "score_std": None if s.size < 2 else round(float(np.std(s, ddof=1)), 6),
        "shrinkage": shrink,
        "pbo": pbo_res,
        "affects_gate": False,
        "note": ("CPCV 只产出参考读数；是否引入过拟合概率下限属 T17.4 人工检查点。"
                 "收缩指标为一阶近似，不冒充 Sharpe DSR 精确解。"),
    }
