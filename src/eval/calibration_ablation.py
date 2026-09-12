"""校准层消融对照（S19 / H4，T19.4 决策证据）。

问题从哪来：
  T19.1/T19.2 已经把校准层的**校准质量**证据做出来了：ECE 0.1042 → 0.0025、
  Brier 0.2650 → 0.2498。但 T19.4 要定的是「校准层要不要进**主推理链路**」——
  这件事的判据不是 Brier/ECE，而是**决策读数**：以校准后概率替代原始概率后，
  一整套下游产物（阈值扫描曲线上的命中率 / IC / 覆盖率）到底变好还是变坏？

  这两件事可以完全脱节：概率更准 ≠ 按它的方向做单更准。校准器是单调映射，
  全样本口径下**方向命中率恒等**（`sign(σ(p) - 0.5) == sign(p - 0.5)`），
  真正的差异只会出现在**阈值子集**口径——而阈值子集正是 T15.3 想让门禁依赖的东西。

本模块做什么（T19.4 消融）：
  1. **同一拟合、三条腿**：一次拟合 LightGBM，产出
     ``base``（原始概率）/ ``platt`` / ``isotonic`` 三条腿的
     复验段概率。三条腿共享**同一个**分类器与同一段校准样本——
     只变"概率是否被校准"这一个变量，与 S12 label-ab / S13 qlib-ab 同款对照纪律；
  2. **决策读数逐指标并排**：门禁点读数（clf=no_call 的命中率 + 覆盖率）、
     阈值扫描曲线（逐阈值 命中率 / IC / 覆盖率 / 样本数）、AUC（排序不变性复核）；
  3. **保守判定 + 负面读数如实入库**：逐指标标 ``improved`` / ``degraded`` /
     ``unchanged`` / ``insufficient``，一升一降不择优、不合成总分；
  4. 落盘 ``reports/calibration/calibration_ablation.json``。

本模块**不做什么**（边界比功能更重要）：
  - **不改主推理链路**：``model.calibration.enabled`` 缺省 false，
    ``affects_gate`` 恒为 False，``strategy_gate`` 逐字段零变更；
  - **不改现行阈值**：阈值曲线只作对照呈现，不选点、不推荐；
  - **不把「校准更准」写成「决策更好」**：两者是不同命题，本报告分开陈述；
  - **不猜**：样本不足 / 校准器不可用 / 指标缺失一律 ``available=false`` + 原因。

无前视说明：
  复用 ``calibrate_and_evaluate`` 的三段式语义：训练段 < 校准段 < 复验段。
  LightGBM 只在**训练段**拟合，校准器只在**校准段**拟合，全部读数只在
  **复验段**（从未参与训练与校准）产生。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.eval import probability_calibration as pc

logger = logging.getLogger(__name__)

REPORT_NAME = "calibration_ablation.json"
DEFAULT_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
# 门禁点：与 strategy_gate 现行放行口径同源（方向命中率 > 0.5 的中性带）
GATE_NEUTRAL = 0.5
MIN_SAMPLES_PER_ROW = 50

LEG_BASE = "base"
LEG_PLATT = "platt"
LEG_ISOTONIC = "isotonic"

_TAGS = ("improved", "degraded", "unchanged", "insufficient")


def _num(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return default if v != v else v


def _r(x: Optional[float], ndigits: int = 6) -> Optional[float]:
    return None if x is None else round(float(x), ndigits)


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _tie(p: np.ndarray) -> np.ndarray:
    """平局样本：概率恰为 0.5（或 NaN）——命中率口径里属中性带。"""
    pn = np.nan_to_num(np.asarray(p, dtype=float), nan=0.5)
    return np.abs(pn - GATE_NEUTRAL) <= 1e-12


def hit_rate_clf(proba: Sequence[float], fwd_ret: Sequence[float]) -> Dict[str, Any]:
    """**同源门禁口径**的命中率：平局样本计入分母但不算命中。

    现行门禁吃的是"判对/判错"，因此必须把"没方向"（p == 0.5）如实算作不命中，
    而不是把它们从分母里剔掉——后者会系统性抬高命中率。
    """
    p = np.asarray(proba, dtype=float)
    r = np.asarray(fwd_ret, dtype=float)
    n = int(min(p.size, r.size))
    if n <= 0:
        return {"available": False, "reason": "no_samples", "samples": 0}
    p, r = p[:n], r[:n]
    pn = np.nan_to_num(p, nan=GATE_NEUTRAL)
    rn = np.nan_to_num(r, nan=0.0)
    ties = _tie(pn)
    correct = np.sign(pn - GATE_NEUTRAL) == np.sign(rn)
    hit = float(np.mean(correct & ~ties))
    return {
        "available": True,
        "samples": n,
        "hit_rate": _r(hit),
        "tie_samples": int(ties.sum()),
        "tie_share": _r(float(ties.mean())),
        "coverage": _r(1.0 - float(ties.mean())),
        "rule": "sign(p−0.5)==sign(fwd_ret) 且 p≠0.5；平局计入分母不算命中",
    }


def _auc(proba: np.ndarray, y_true: np.ndarray) -> Optional[float]:
    """AUC（Mann-Whitney 形式，不依赖 sklearn 版本）。标签退化 → None（不猜）。"""
    p = np.nan_to_num(np.asarray(proba, dtype=float), nan=GATE_NEUTRAL)
    y = np.asarray(y_true, dtype=float)
    n = int(min(p.size, y.size))
    if n <= 0:
        return None
    p, y = p[:n], y[:n]
    pos, neg = p[y > 0.5], p[y <= 0.5]
    if pos.size == 0 or neg.size == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, dtype=float)
    ranks[order] = np.arange(1, order.size + 1, dtype=float)
    # 平均秩（处理并列）
    allv = np.concatenate([pos, neg])
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size, dtype=float)
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    ranks = avg[inv]
    r_pos = float(ranks[:pos.size].sum())
    return float((r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def criterion_row(name: str, base: Optional[float], leg: Optional[float],
                  higher_is_better: bool = True,
                  eps: float = 1e-9,
                  note: str = "") -> Dict[str, Any]:
    """逐指标对照（**不做加权总分、不择优**）。

    一升一降如实各记各的；只有单一指标时按方向判 improved/degraded/unchanged。
    """
    out: Dict[str, Any] = {
        "criterion": name,
        "base": _r(base),
        "leg": _r(leg),
        "higher_is_better": bool(higher_is_better),
        "note": note,
        "available": base is not None and leg is not None,
    }
    if not out["available"]:
        out["tag"] = "insufficient"
        out["delta"] = None
        return out
    delta = float(leg) - float(base)
    out["delta"] = _r(delta)
    direction = delta if higher_is_better else -delta
    if direction > eps:
        out["tag"] = "improved"
    elif direction < -eps:
        out["tag"] = "degraded"
    else:
        out["tag"] = "unchanged"
    return out


def _gate_point(proba: np.ndarray, fwd_ret: np.ndarray,
                y_true: np.ndarray) -> Dict[str, Any]:
    """门禁点读数（clf=no_call 口径）：命中率 + 覆盖率 + AUC。"""
    from src.inference.ic import spearman_ic

    hr = hit_rate_clf(proba, fwd_ret)
    ic = None
    if hr.get("available") and int(hr["samples"]) >= 2:
        ic = _r(float(spearman_ic(np.asarray(proba, dtype=float),
                                  np.asarray(fwd_ret, dtype=float))))
    row = dict(hr)
    row["ic"] = ic
    row["auc"] = _r(_auc(np.asarray(proba, dtype=float),
                         np.asarray(y_true, dtype=float)))
    return row


def _threshold_grid(proba: np.ndarray, fwd_ret: np.ndarray,
                    grid: Sequence[float], min_samples: int) -> Dict[str, Any]:
    """按置信度阈值扫「命中率 / IC / 覆盖率」（净值口径，不择优）。"""
    from src.eval.confidence_curve import confidence_from_proba
    from src.inference.ic import spearman_ic

    c = confidence_from_proba(proba)
    rows: List[Dict[str, Any]] = []
    for thr in grid:
        thr = float(thr)
        mask = c >= thr
        kept = int(mask.sum())
        row: Dict[str, Any] = {
            "threshold": thr,
            "samples": kept,
            "coverage": _r(kept / len(proba)) if len(proba) else 0.0,
        }
        if kept < max(int(min_samples), 2):
            row.update({"available": False, "reason": "insufficient_samples",
                        "hit_rate": None, "ic": None})
        else:
            hr = hit_rate_clf(proba[mask], fwd_ret[mask])
            row.update({
                "available": True,
                "hit_rate": hr.get("hit_rate"),
                "ic": _r(float(spearman_ic(np.asarray(proba[mask], dtype=float),
                                           np.asarray(fwd_ret[mask], dtype=float)))),
            })
        rows.append(row)
    return {"kind": "confidence_curve_ablation", "rows": rows,
            "grid": [float(g) for g in grid], "affects_gate": False}


# ----------------------------------------------------------------------
# 主入口：构建消融报告（纯函数，输入概率即可单测）
# ----------------------------------------------------------------------
def build_ablation_report(legs: Dict[str, Dict[str, Any]],
                          meta: Optional[Dict[str, Any]] = None,
                          grid: Sequence[float] = DEFAULT_GRID,
                          min_samples: int = MIN_SAMPLES_PER_ROW) -> Dict[str, Any]:
    """把三条腿（base/platt/isotonic）的复验段概率压成对照报告。

    ``legs`` 形如::

        {"base": {"proba": [...], "y_true": [...], "fwd_ret": [...]},
         "platt": {"proba": [...], ...}, ...}

    所有腿必须共享同一批复验段样本（同一 y_true / fwd_ret）——否则对照无效，
    此时如实标记 ``aligned=false`` 并不产结论。
    """
    available_legs = [k for k, v in legs.items()
                      if v and v.get("available") and v.get("proba") is not None]
    out: Dict[str, Any] = {
        "kind": "calibration_ablation",
        "generated_at": _now(),
        "evidence_for": "T19.4（校准层是否进主推理链路）",
        "legs": {},
        "criteria": {},
        "aligned": False,
        "available": False,
        "affects_gate": False,
        "grid": [float(g) for g in grid],
        "note": ("同一拟合、同一复验段，只变「概率是否被校准」；"
                 "逐指标对照不合成总分、不择优；是否切换属 T19.4 人工检查点。"),
        "meta": dict(meta or {}),
    }
    if LEG_BASE not in available_legs:
        out["reason"] = "缺少 base 腿（原始概率），无法对照"
        return out

    n_min = min(int(len(legs[k]["proba"])) for k in available_legs)
    if n_min <= 0:
        out["reason"] = "复验段无样本"
        return out

    # 对齐校验：所有腿的 y_true / fwd_ret 必须一致（防拿两批样本对照）
    ref = legs[LEG_BASE]
    ref_y = np.asarray(ref.get("y_true"), dtype=float)[:n_min]
    ref_r = np.asarray(ref.get("fwd_ret"), dtype=float)[:n_min]
    for k in available_legs:
        y = np.asarray(legs[k].get("y_true"), dtype=float)[:n_min]
        r = np.asarray(legs[k].get("fwd_ret"), dtype=float)[:n_min]
        if not (np.allclose(y, ref_y, equal_nan=True)
                and np.allclose(r, ref_r, equal_nan=True)):
            out["reason"] = (f"{k} 腿的 y_true/fwd_ret 与 base 不一致："
                             "样本未对齐，对照无效（不猜）")
            return out
    out["aligned"] = True
    out["verify_samples"] = n_min

    for k in available_legs:
        p = np.asarray(legs[k]["proba"], dtype=float)[:n_min]
        out["legs"][k] = {
            "available": True,
            "samples": n_min,
            "gate_point": _gate_point(p, ref_r, ref_y),
            "threshold_curve": _threshold_grid(p, ref_r, grid, min_samples),
            "probability_metrics": {
                "brier": _r(pc.brier_score(p, ref_y)),
                "ece": _r(pc.expected_calibration_error(p, ref_y)),
            },
        }
    out["available"] = len(out["legs"]) >= 2
    if not out["available"]:
        out["reason"] = "可用腿不足 2 条（校准器不可用），不构成对照"
        return out

    # ---- 逐指标对照：base 为参照，其余腿逐条比 ----
    base_gate = out["legs"][LEG_BASE]["gate_point"]
    base_prob = out["legs"][LEG_BASE]["probability_metrics"]
    base_rows = {r["threshold"]: r
                 for r in out["legs"][LEG_BASE]["threshold_curve"]["rows"]}
    for k in [x for x in available_legs if x != LEG_BASE]:
        leg = out["legs"][k]
        gate = leg["gate_point"]
        crit: List[Dict[str, Any]] = [
            criterion_row("门禁点命中率（clf=no_call）", base_gate.get("hit_rate"),
                          gate.get("hit_rate"),
                          note="与 strategy_gate 同源口径；平局计入分母"),
            criterion_row("门禁点 IC", base_gate.get("ic"), gate.get("ic")),
            criterion_row("AUC（方向排序能力）", base_gate.get("auc"), gate.get("auc"),
                          note="校准是单调映射：AUC 不应变化，变了说明装配有问题"),
            criterion_row("Brier（越低越好）", base_prob.get("brier"),
                          leg["probability_metrics"].get("brier"),
                          higher_is_better=False),
            criterion_row("ECE（越低越好）", base_prob.get("ece"),
                          leg["probability_metrics"].get("ece"),
                          higher_is_better=False),
        ]
        # 阈值子集口径（T15.3 关心的正是这个）：逐阈值命中率/IC
        leg_rows = {r["threshold"]: r for r in leg["threshold_curve"]["rows"]}
        hit_pairs = [(base_rows[t].get("hit_rate"), leg_rows[t].get("hit_rate"))
                     for t in sorted(base_rows)
                     if t in leg_rows
                     and base_rows[t].get("available") and leg_rows[t].get("available")]
        if hit_pairs:
            b_hits = [a for a, _ in hit_pairs]
            l_hits = [b for _, b in hit_pairs]
            crit.append(criterion_row(
                "阈值子集命中率（各阈值均值）",
                float(np.mean(b_hits)), float(np.mean(l_hits)),
                note=f"{len(hit_pairs)} 个可用阈值行的均值对照；逐阈值明细见 threshold_curve"))

        tags = [c["tag"] for c in crit if c["tag"] != "insufficient"]
        verdict = "insufficient"
        if tags:
            if all(t == "improved" for t in tags):
                verdict = "improved"
            elif all(t == "degraded" for t in tags):
                verdict = "degraded"
            elif any(t == "improved" for t in tags) and any(t == "degraded" for t in tags):
                verdict = "mixed"
            else:
                verdict = "no_material_change"
        out["criteria"][k] = {
            "leg": k,
            "criteria": crit,
            "tag_counts": {t: tags.count(t) for t in _TAGS},
            "verdict": verdict,
            "reason": {
                "improved": "全部可用指标同向变好（含门禁点口径）",
                "degraded": "全部可用指标同向变差（如实入库，不粉饰）",
                "mixed": "存在指标一升一降：不构成改善结论，不择优",
                "no_material_change": "全部可用指标无实质变化",
                "insufficient": "可用指标不足，不猜",
            }[verdict],
        }
    return out


def build_leg_payload(proba: Sequence[float], y_true: Sequence[int],
                      fwd_ret: Sequence[float]) -> Dict[str, Any]:
    """单条腿的输入封装（概率 + 同批复验段标签/收益）。"""
    return {"available": True, "proba": list(proba),
            "y_true": list(y_true), "fwd_ret": list(fwd_ret)}


def report_path(config: Optional[Dict[str, Any]] = None) -> Path:
    """报告路径（复用 S16 已定的 ``reports/calibration/`` 目录，不另立口径）。"""
    from src.eval.conformal_probability import calibration_dir

    return calibration_dir(config) / REPORT_NAME
