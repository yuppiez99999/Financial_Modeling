"""区间口径 vs 概率距离口径的对照 + 多时段滚动保留期复验（S16 / H1，T16.2 / T16.3）。

问题从哪来：
  T15.3 的置信度门槛建立在 ``|p−0.5|×2``（概率距离）上，且**只有一个保留期**。
  两处遗留：
  1. 口径未对照：区间宽度换算的置信分（``confidence_from_interval``，G5 已
     抽象但未实装）与概率距离口径到底哪个更能筛出高命中子集？**同数据同折**
     比一次才知道；
  2. 时段未复验："thr↑→命中率↑"只在单一时段验证过，跨时段是否稳定未知 ——
     这正是 T15.3 自己写的下一轮待办。

本模块做什么：
  - **口径对照**（T16.2）：同一份 walk-forward 测试折样本上，分别用
    「概率距离」与「保形区间宽度」两种置信分做阈值扫描，逐阈值对比
    覆盖率 / 命中率 / IC，给出并排表与结论（哪种更能分离高命中子集）；
  - **多时段滚动保留期**（T16.3）：把复验段按**时间顺序**切成若干连续子段
    （季度口径），每段独立跑同一套阈值扫描，看「命中率优势」是否跨段稳定；
    输出逐段读数 + 稳定性判定（一致 / 部分一致 / 不一致）。

本模块**不做什么**（边界比功能重要）：
  - **不改门禁**：``affects_gate`` 恒为 False，不写配置；
  - **不代选口径/阈值**：只给对照读数与稳定性判定，取舍属 T16.4 人工检查点；
  - **不做统计显著性声明**：分段读数样本量小，报告显式标注"仅呈现，不作
    达标证据"，避免把选择自由度洗成结论；
  - **不猜**：任一段样本不足 → 该段 ``available=false`` + 原因。

无前视说明：
  分段只沿**测试折时间顺序**切分，每段内部阈值扫描不跨段共享统计量；
  段与段之间不重叠，任何段都不含训练折信息。
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.eval.confidence_curve import (
    confidence_from_interval,
    confidence_from_proba,
    sweep_confidence,
)

logger = logging.getLogger(__name__)

DEFAULT_GRID = (0.0, 0.1, 0.2, 0.3, 0.4)
MIN_SEGMENT_SAMPLES = 100
MIN_STABLE_SEGMENTS = 2


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _available_rows(curve: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in ((curve or {}).get("rows") or [])
            if isinstance(r, dict) and r.get("available")]


def compare_confidence_sources(
    proba: Sequence[float],
    fwd_ret: Sequence[float],
    *,
    lower: Optional[Sequence[float]] = None,
    upper: Optional[Sequence[float]] = None,
    mid: Optional[Sequence[float]] = None,
    grid: Sequence[float] = DEFAULT_GRID,
    min_samples: int = 50,
    horizon_days: Optional[int] = None,
) -> Dict[str, Any]:
    """同数据同折对照两种置信度口径（T16.2，纯函数）。

    - 口径 A：``|p−0.5|×2``（现行，概率距离）；
    - 口径 B：保形/概率区间宽度换算（``confidence_from_interval``，需区间）。

    区间缺失时不编造口径 B 的读数 —— 该口径标 ``available=false`` 并给原因。
    """
    p = np.asarray(proba, dtype=float)
    r = np.asarray(fwd_ret, dtype=float)
    n = int(min(p.size, r.size))
    p, r = p[:n], r[:n]
    if n == 0:
        return {"kind": "confidence_source_ab", "available": False,
                "reason": "无样本", "affects_gate": False, "rows": []}

    curve_a = sweep_confidence(p, r, confidence=None, grid=grid,
                               min_samples=min_samples)

    width_available = all(x is not None for x in (lower, upper, mid))
    curve_b: Optional[Dict[str, Any]] = None
    if width_available:
        conf_b = confidence_from_interval(lower, upper, mid)  # type: ignore[arg-type]
        if conf_b.size >= n:
            curve_b = sweep_confidence(p, r, confidence=conf_b[:n], grid=grid,
                                       min_samples=min_samples)

    by_thr_a = {row["threshold"]: row for row in curve_a["rows"]}
    by_thr_b = ({row["threshold"]: row for row in curve_b["rows"]}
                if curve_b else {})
    rows: List[Dict[str, Any]] = []
    for thr in [row["threshold"] for row in curve_a["rows"]]:
        ra = by_thr_a.get(thr, {})
        rb = by_thr_b.get(thr)
        row: Dict[str, Any] = {
            "threshold": thr,
            "proba_distance": {
                "available": bool(ra.get("available")),
                "coverage": ra.get("coverage"),
                "hit_rate": ra.get("hit_rate"),
                "ic": ra.get("ic"),
            },
            "interval_width": {
                "available": bool((rb or {}).get("available")),
                "coverage": (rb or {}).get("coverage"),
                "hit_rate": (rb or {}).get("hit_rate"),
                "ic": (rb or {}).get("ic"),
            },
            "hit_rate_delta_interval_minus_proba": None,
        }
        if row["proba_distance"]["available"] and row["interval_width"]["available"]:
            row["hit_rate_delta_interval_minus_proba"] = round(
                float(rb["hit_rate"]) - float(ra["hit_rate"]), 6)
        rows.append(row)

    # 结论：只在两侧都有可用读数时下"哪种口径更分离高命中子集"
    verdict = "unavailable"
    detail = "区间口径不可用（缺区间/区间与样本不对齐），不做口径结论"
    deltas = [x["hit_rate_delta_interval_minus_proba"] for x in rows
              if x["hit_rate_delta_interval_minus_proba"] is not None]
    if deltas:
        mean_delta = float(np.mean(deltas))
        pos = sum(1 for d in deltas if d > 0)
        if mean_delta > 0.005 and pos > len(deltas) / 2:
            verdict = "interval_more_separative"
            detail = f"区间口径在 {pos}/{len(deltas)} 个阈值上命中率更高（均差 {mean_delta:+.4f}）"
        elif mean_delta < -0.005 and (len(deltas) - pos) > len(deltas) / 2:
            verdict = "proba_distance_more_separative"
            detail = (f"概率距离口径更优：区间口径仅在 {pos}/{len(deltas)} 个阈值更高"
                      f"（均差 {mean_delta:+.4f}）")
        else:
            verdict = "no_material_difference"
            detail = f"两口径无实质差异（命中率均差 {mean_delta:+.4f}）"

    return {
        "kind": "confidence_source_ab",
        "generated_at": _now(),
        "horizon_days": horizon_days,
        "available": True,
        "samples": n,
        "interval_provided": bool(width_available),
        "verdict": verdict,
        "verdict_detail": detail,
        "rows": rows,
        "grid_proba": list(grid),
        "grid_interval": list(grid) if curve_b else [],
        "proba_observation": curve_a.get("observation"),
        "interval_observation": (curve_b or {}).get("observation"),
        "affects_gate": False,
        "note": ("同数据同折对照；读数属选择自由度，未经多重比较校正不得"
                 "直接作为门禁达标证据。口径取舍属 T16.4 人工检查点。"),
    }


def split_time_segments(n: int, segments: int = 4,
                        min_samples: int = MIN_SEGMENT_SAMPLES) -> List[Dict[str, Any]]:
    """把**时间顺序**样本切成连续子段（T16.3 滚动保留期口径）。

    不重叠、不跨段：第 i 段 = ``[i*size, (i+1)*size)``。样本不足时返回空列表
    （由调用方标不可用，不猜）。
    """
    n = int(n)
    k = int(segments)
    if n <= 0 or k <= 0:
        return []
    if n < min_samples * k:
        return []
    size = n // k
    out: List[Dict[str, Any]] = []
    for i in range(k):
        lo = i * size
        hi = n if i == k - 1 else (i + 1) * size
        if hi - lo < min_samples:
            return []
        out.append({"index": i, "start": lo, "end": hi, "samples": hi - lo})
    return out


def rolling_holdout_verify(
    proba: Sequence[float],
    fwd_ret: Sequence[float],
    *,
    segments: int = 4,
    thresholds: Sequence[float] = (0.0, 0.2, 0.3),
    min_hit_rate: float = 0.52,
    min_samples: int = MIN_SEGMENT_SAMPLES,
    horizon_days: Optional[int] = None,
) -> Dict[str, Any]:
    """多时段滚动保留期复验：逐子段检查「thr↑ → 命中率↑」是否稳定（T16.3）。

    只在**尾部子段**（复验口径）上读数，逐段给出每个阈值的命中率与覆盖率；
    稳定性判定基于"命中率随阈值单调不减"的段数占比，**不做显著性声明**。
    """
    p = np.asarray(proba, dtype=float)
    r = np.asarray(fwd_ret, dtype=float)
    n = int(min(p.size, r.size))
    p, r = p[:n], r[:n]
    segs = split_time_segments(n, segments=segments, min_samples=min_samples)
    if not segs:
        return {
            "kind": "rolling_holdout_verify", "generated_at": _now(),
            "horizon_days": horizon_days, "available": False,
            "reason": (f"样本不足：n={n}，需 ≥{min_samples}×{segments}="
                       f"{min_samples * segments}，不猜"),
            "segments": [], "affects_gate": False,
            "note": "多时段复验仅作呈现，不作达标证据（T16.4 人工检查点）",
        }

    thrs = sorted(float(t) for t in thresholds)
    out_segs: List[Dict[str, Any]] = []
    monotone_flags: List[bool] = []
    for seg in segs:
        ps, rs = p[seg["start"]:seg["end"]], r[seg["start"]:seg["end"]]
        curve = sweep_confidence(ps, rs, grid=thrs, min_samples=1)
        by_thr = {row["threshold"]: row for row in curve["rows"]}
        readings = []
        for thr in thrs:
            row = by_thr.get(thr, {})
            readings.append({
                "threshold": thr,
                "available": bool(row.get("available")),
                "coverage": row.get("coverage"),
                "samples": row.get("samples"),
                "hit_rate": row.get("hit_rate"),
                "ic": row.get("ic"),
                "meets_min_hit_rate": (
                    None if not row.get("available")
                    else bool(float(row.get("hit_rate") or 0.0) >= float(min_hit_rate))
                ),
            })
        avail = [x for x in readings if x["available"]]
        monotone = None
        if len(avail) >= 2:
            hits = [float(x["hit_rate"]) for x in avail]
            monotone = bool(all(b >= a - 0.005 for a, b in zip(hits, hits[1:])))
            monotone_flags.append(monotone)
        out_segs.append({
            "index": seg["index"], "samples": seg["samples"],
            "readings": readings,
            "hit_rate_monotone_nondecreasing": monotone,
        })

    usable = [f for f in monotone_flags]
    if not usable:
        stability = "unavailable"
        detail = "各段可用读数不足 2 个阈值，无法判定稳定性"
    else:
        share = sum(1 for f in usable if f) / len(usable)
        if share >= 0.75:
            stability = "stable"
            detail = f"{sum(1 for f in usable if f)}/{len(usable)} 段命中率随阈值单调不减"
        elif share >= 0.5:
            stability = "partially_stable"
            detail = f"仅 {sum(1 for f in usable if f)}/{len(usable)} 段单调不减（部分一致）"
        else:
            stability = "unstable"
            detail = f"仅 {sum(1 for f in usable if f)}/{len(usable)} 段单调不减（跨时段不一致）"

    return {
        "kind": "rolling_holdout_verify",
        "generated_at": _now(),
        "horizon_days": horizon_days,
        "available": True,
        "samples": n,
        "segments_requested": int(segments),
        "segments_used": len(out_segs),
        "thresholds": thrs,
        "min_hit_rate": float(min_hit_rate),
        "stability": stability,
        "stability_detail": detail,
        "segments": out_segs,
        "affects_gate": False,
        "note": ("逐子段读数样本量小，仅呈现跨时段一致性，不作达标证据；"
                 "是否切换 thr 属 T15.3 / T16.4 人工检查点。"),
    }
