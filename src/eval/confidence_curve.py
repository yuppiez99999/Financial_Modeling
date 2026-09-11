"""置信度阈值曲线（S15 / G5，T15.2）：只对高置信样本给信号。

为什么需要（Issue #29 集成方案 G5 验收口径）：
  门禁卡在"全样本命中率 ~50%"，但一个被反复观察到的现象是：**模型在自己
  最有把握的子集上可能显著更准**。G5 的验证口径是「置信度 ≥X 才输出信号」
  （与 Q4 风控建议 `withheld` 语义一致）：把 X 从 0 扫到上限，画出
  「覆盖率 × 命中率 × IC」的完整曲线，让 T15.3 人工检查点可以基于
  覆盖/精度 trade-off 做决策，而不是拍脑袋定阈值。

  neuralforecast 的概率区间（quantile 预测区间宽度 → 置信度）就是这一
  机制的一种实现：区间越窄，模型越有把握。本模块把该机制抽象为
  **任一 [0,1] 置信度分数**（概率区间宽度的归一化、或 LightGBM 概率到
  0.5 的距离都满足），neuralforecast 集成时只需把区间宽度换算成置信分
  喂进来（见 `confidence_from_interval`），不必另写一条链路。

本模块做什么：
  - 输入：模型打分（概率）、未来真实收益、每样本置信度分数；
  - 沿阈值网格扫描：覆盖率（保留多少样本）、命中率（中性带 0.5 同门禁口径）、
    IC（仅保留样本内）、样本数；
  - 输出逐阈值行 + 推荐观察点，落盘 `reports/confidence_curve_<horizon>d.json`。

本模块**不做什么**（边界比功能重要）：
  - **不改门禁**：`affects_gate` 恒为 False，不自动选阈值；
  - **不挑曲线上的好点当结论**：阈值扫描本身是选择自由度，任何"某阈值下
    命中率 55%"的读数都必须结合覆盖率和登记的试验次数（S13）解读，
    未经多重比较校正不得引用为达标证据；
  - **不在训练折上扫阈值再报测试折**：曲线只是把 trade-off **呈现**出来，
    不构成任何"已验证可提升"的声明；
  - **不猜**：样本不足的阈值行 `available=false`。

统计口径备注：
  - 命中率与门禁同源 `src.inference.ic.hit_rate`（分数去 0.5 中性化后比方向）；
  - IC 用 Spearman（同源 `spearman_ic`）；
  - 置信度口径：`|p - 0.5| × 2`（概率自 0.5 的距离归一化），或
    概率区间宽度的单调递减函数（`confidence_from_interval`）。
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.inference.ic import hit_rate, spearman_ic

logger = logging.getLogger(__name__)

MIN_SAMPLES_PER_BIN = 50          # 低于此样本数的阈值行不给指标（不猜）
DEFAULT_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def confidence_from_proba(proba: Sequence[float]) -> np.ndarray:
    """概率 → 置信度：|p − 0.5| × 2（0=无观点，1=极端自信）。"""
    p = np.asarray(proba, dtype=float)
    return np.clip(np.abs(p - 0.5) * 2.0, 0.0, 1.0)


def confidence_from_interval(
    lower: Sequence[float],
    upper: Sequence[float],
    mid: Sequence[float],
) -> np.ndarray:
    """概率区间 → 置信度（neuralforecast 口径）。

    区间宽度相对中位数水平归一化后取补：区间越窄 → 置信度越高。
    与 `withheld` 语义一致：低置信 = 不给信号。
    """
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    base = np.asarray(mid, dtype=float)
    base = np.where(np.abs(base) < 1e-9, 1e-9, base)
    width = np.maximum(hi - lo, 0.0) / np.abs(base)
    return np.clip(1.0 - width, 0.0, 1.0)


def _neutralized_direction(proba: np.ndarray) -> np.ndarray:
    """概率去 0.5 中性化：>0 正、<0 负（供 hit_rate 的中性带口径复用）。"""
    return proba - 0.5


def sweep_confidence(
    proba: Sequence[float],
    fwd_ret: Sequence[float],
    confidence: Optional[Sequence[float]] = None,
    grid: Sequence[float] = DEFAULT_GRID,
    min_samples: int = MIN_SAMPLES_PER_BIN,
) -> Dict[str, Any]:
    """沿置信度阈值扫描「覆盖率 / 命中率 / IC」曲线。"""
    p = np.asarray(proba, dtype=float)
    r = np.asarray(fwd_ret, dtype=float)
    n = min(len(p), len(r))
    p, r = p[:n], r[:n]
    if confidence is None:
        c = confidence_from_proba(p)
    else:
        c = np.asarray(confidence, dtype=float)[:n]

    scores = _neutralized_direction(p)
    rows: List[Dict[str, Any]] = []
    for thr in grid:
        thr = float(thr)
        mask = c >= thr
        kept = int(mask.sum())
        row: Dict[str, Any] = {
            "threshold": thr,
            "coverage": round(kept / n, 4) if n else 0.0,
            "samples": kept,
        }
        if kept < max(min_samples, 2):
            row.update({"available": False,
                        "reason": "insufficient_samples"})
        else:
            row.update({
                "available": True,
                "hit_rate": round(hit_rate(scores[mask], r[mask]), 4),
                "ic": round(spearman_ic(scores[mask], r[mask]), 4),
            })
        rows.append(row)

    # 观察点：覆盖率 ≥ 30% 且可用的行里命中率最高（只作呈现，不据此选阈值）
    candidates = [row for row in rows
                  if row.get("available") and row["coverage"] >= 0.3]
    observation = None
    if candidates:
        best = max(candidates, key=lambda row: row["hit_rate"])
        observation = {
            "threshold": best["threshold"],
            "hit_rate": best["hit_rate"],
            "ic": best["ic"],
            "coverage": best["coverage"],
            "note": "呈现用观察点，非推荐阈值；选阈值为 T15.3 人工检查点",
        }

    return {
        "kind": "confidence_curve",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_samples": int(n),
        "confidence_source": "provided" if confidence is not None else "proba_distance",
        "min_samples_per_bin": int(min_samples),
        "rows": rows,
        "observation": observation,
        "affects_gate": False,
        "note": (
            "阈值扫描为选择自由度：任何单阈值读数未经多重比较校正不得作为"
            "门禁达标证据（见 S13 试验登记）。是否落地置信度门槛属 T15.3 人工检查点。"
        ),
    }
