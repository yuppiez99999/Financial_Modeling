"""置信度子集门禁（S15 / G5，T15.3）：把门禁从「全样本命中率」升级为
「置信度 ≥thr 子集命中率 + 覆盖率下限」双指标。

问题从哪来（S15 一轮的最终证据）：
  G5 结束时唯一在**独立保留期**验证过的改善机制，就是这条：
  在保留期（后 30%，从未参与任何训练 / 扫描 / 超参搜索）上，
  「thr↑ → 子集命中率↑」单调成立且 IC 同向，候选区间 thr ∈ [0.2, 0.3]。

  但把它落成门禁，动的是 `strategy_gate` 的**结构**（判据从"全样本命中率"
  变成"子集命中率 + 覆盖率"），属于产品口径变更 —— 按仓库既有纪律
  （S11 / S12 / S13 同款），**必须人工签字**，任何自动流程不得代签。

本模块做什么：
  1. **双指标判定**（纯函数）：给一条置信度曲线 + 一个阈值，
     用「子集命中率 ≥ min_hit_rate」**且**「覆盖率 ≥ min_coverage」两条腿
     同时过线才算候选达标 —— 只有一条腿过线一律判不合格：
     - 只过命中率、覆盖率过低 → 信号量收缩到不可用（"只留 3 个样本 100%"陷阱）；
     - 只过覆盖率、命中率不够   → 就是现行全样本口径，没有改善。
  2. **候选区间扫描**：在 thr ∈ [thr_min, thr_max] 网格上找出**同时**满足
     双指标的阈值区间，并给出每个阈值的双指标读数。
  3. **决策单**（`build_decision_record`）：把「是否改门禁结构」落成
     可审计记录 —— 提议、双指标证据、三种结论（approve / reject / defer）、
     人工确认状态与理由。**没有 `decided_by` 就永远是 pending**。
  4. **独立保留期优先**：证据默认取保留期复验报告（未参与扫描）；
     walk-forward 曲线只作补充呈现，不得替代保留期证据。

本模块**不做什么**（边界比功能重要）：
  - **不改门禁**：`affects_gate` 恒为 False，`strategy_gate` 放行结论逐字段不变；
    本模块只产出"是否值得改"的决策材料；
  - **不自动选阈值**：候选区间是**范围**不是单点；落在区间里的任一 thr 都
    必须由人工在决策单里显式挑定（`chosen_threshold`），本模块不代选；
  - **不签字**：无 `decided_by` → 结论最多是 `defer`，`status` 恒 `pending`；
  - **不猜**：样本不足 / 报告缺失 / 阈值无可用行 → 如实 `available=false` 并给原因；
  - **不做无依据外推**：保留期是单一时段（最近 ~1.5 年），本模块显式标注
    `single_period_warning`，不把保留期水位当常态水平。

无前视说明：
  本模块只消费既有报告（`reports/confidence_holdout_verify.json` /
  `reports/confidence_curve_<h>d.json`）里**已对齐**的指标，
  不重训模型、不重新构造目标、不接触价格序列，因此不引入任何前视。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

REPORT_NAME = "confidence_gate_decision.json"
DEFAULT_HOLDOUT_NAME = "confidence_holdout_verify.json"

# 决策结论（defer 是默认值，不是失败态 —— 与 S11 horizon_decision 同款语义）
VERDICT_APPROVE = "approve"
VERDICT_REJECT = "reject"
VERDICT_DEFER = "defer"

STATUS_PENDING = "pending"
STATUS_CONFIRMED = "confirmed"
STATUS_STALE = "stale"

# 候选阈值区间（Issue #29 指令给定：thr ∈ [0.2, 0.3]）
DEFAULT_THR_MIN = 0.2
DEFAULT_THR_MAX = 0.3

# 双指标默认口径（与 strategy_gate.min_hit_rate 同源；覆盖率为下限）
DEFAULT_MIN_HIT_RATE = 0.52
DEFAULT_MIN_COVERAGE = 0.05
DEFAULT_MIN_SAMPLES = 50


def _cfg(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """读 `strategy_gate.confidence_gate` 段（缺省空字典）。"""
    gate = ((config or {}).get("strategy_gate", {}) or {})
    if not isinstance(gate, dict):
        return {}
    seg = gate.get("confidence_gate", {}) or {}
    return seg if isinstance(seg, dict) else {}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default  # 过滤 NaN


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ----------------------------------------------------------------------
# 双指标判定（纯函数）
# ----------------------------------------------------------------------
def evaluate_threshold_row(row: Dict[str, Any], min_hit_rate: float,
                           min_coverage: float,
                           min_samples: int = DEFAULT_MIN_SAMPLES) -> Dict[str, Any]:
    """对**一条**阈值行做双指标判定（纯函数，不读配置、不落盘）。

    两条腿**都**过线才算 `passed=True`；任一条不过，`blocked_by` 里如实写明是哪条。
    """
    thr = round(_num(row.get("threshold")), 4)
    samples = int(_num(row.get("samples")))
    coverage = _num(row.get("coverage"))
    available = bool(row.get("available")) and samples >= int(min_samples)
    out: Dict[str, Any] = {
        "threshold": thr,
        "available": available,
        "coverage": round(coverage, 6),
        "samples": samples,
        "hit_rate": _num(row.get("hit_rate")) if available else None,
        "ic": _num(row.get("ic")) if available else None,
        "min_hit_rate": float(min_hit_rate),
        "min_coverage": float(min_coverage),
        "hit_rate_passed": False,
        "coverage_passed": False,
        "passed": False,
        "blocked_by": [],
        "reason": "",
    }
    if not available:
        out["blocked_by"] = ["insufficient_samples"]
        out["reason"] = f"样本不足（{samples} < {min_samples}），不猜"
        return out

    hit = _num(out["hit_rate"])
    out["hit_rate_passed"] = bool(hit >= float(min_hit_rate))
    out["coverage_passed"] = bool(coverage >= float(min_coverage))
    blocked: List[str] = []
    if not out["hit_rate_passed"]:
        blocked.append(f"命中率 {hit:.2%} < {float(min_hit_rate):.2%}")
    if not out["coverage_passed"]:
        blocked.append(f"覆盖率 {coverage:.2%} < {float(min_coverage):.2%}")
    out["blocked_by"] = blocked
    out["passed"] = not blocked
    if out["passed"]:
        out["reason"] = (
            f"双指标同时过线：命中率 {hit:.2%} ≥ {float(min_hit_rate):.2%}，"
            f"覆盖率 {coverage:.2%} ≥ {float(min_coverage):.2%}"
        )
    else:
        out["reason"] = "双指标未同时过线：" + "；".join(blocked)
    return out


def evaluate_curve(curve: Optional[Dict[str, Any]], min_hit_rate: float,
                   min_coverage: float, thr_min: float = DEFAULT_THR_MIN,
                   thr_max: float = DEFAULT_THR_MAX,
                   min_samples: int = DEFAULT_MIN_SAMPLES) -> Dict[str, Any]:
    """对一条置信度曲线做候选区间扫描 + 双指标判定（纯函数）。

    只有落在 `[thr_min, thr_max]` 内、且双指标同时过线的阈值，才算**候选阈值**。
    区间外的行仍会逐行判定（供呈现），但标 `in_candidate_range=false`。
    """
    rows_in = (curve or {}).get("rows") or []
    result: Dict[str, Any] = {
        "available": False,
        "curve_found": bool(curve),
        "curve_kind": str((curve or {}).get("kind") or ""),
        "confidence_source": str((curve or {}).get("confidence_source") or ""),
        "total_samples": int(_num((curve or {}).get("total_samples"))),
        "thresholds": {
            "min_hit_rate": float(min_hit_rate),
            "min_coverage": float(min_coverage),
            "thr_min": float(thr_min),
            "thr_max": float(thr_max),
            "min_samples": int(min_samples),
        },
        "rows": [],
        "candidate_thresholds": [],
        "candidate_range": None,
        "reason": "",
    }
    if not curve or not rows_in:
        result["reason"] = "缺少置信度曲线（先跑 `python main.py confidence`）"
        return result

    evaluated: List[Dict[str, Any]] = []
    for row in rows_in:
        if not isinstance(row, dict):
            continue
        ev = evaluate_threshold_row(row, min_hit_rate, min_coverage,
                                    min_samples=min_samples)
        thr = ev["threshold"]
        ev["in_candidate_range"] = bool(float(thr_min) - 1e-9 <= thr <= float(thr_max) + 1e-9)
        evaluated.append(ev)
    evaluated.sort(key=lambda r: r["threshold"])
    result["rows"] = evaluated
    result["available"] = any(r["available"] for r in evaluated)

    candidates = [r["threshold"] for r in evaluated
                  if r.get("available") and r.get("in_candidate_range") and r.get("passed")]
    result["candidate_thresholds"] = candidates
    if candidates:
        result["candidate_range"] = [min(candidates), max(candidates)]
        result["reason"] = (
            f"候选区间 [{min(candidates)}, {max(candidates)}]（thr ∈ "
            f"[{thr_min}, {thr_max}] 内双指标同时过线的阈值）"
        )
    else:
        result["reason"] = (
            f"thr ∈ [{thr_min}, {thr_max}] 内没有阈值同时满足双指标"
            f"（命中率 ≥ {float(min_hit_rate):.2%} 且覆盖率 ≥ {float(min_coverage):.2%}）"
        )
    return result


# ----------------------------------------------------------------------
# 保留期证据（优先）
# ----------------------------------------------------------------------
def _extract_holdout_curves(holdout: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """从保留期复验报告里按周期抽出曲线（兼容多种字段布局）。

    实采报告结构（`reports/confidence_holdout_verify.json`）通常形如
    ``{"horizons": {"5d": {"rows": [...], ...}}}`` 或 ``{"5d": {...}}``；
    两种都支持，其余情况如实返回空（不猜）。
    """
    if not isinstance(holdout, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for key in ("horizons", "curves", "results"):
        seg = holdout.get(key)
        if isinstance(seg, dict):
            for h, item in seg.items():
                if isinstance(item, dict) and item.get("rows"):
                    out[str(h)] = item
            if out:
                return out
    for h, item in holdout.items():
        if isinstance(item, dict) and item.get("rows") and str(h).endswith("d"):
            out[str(h)] = item
    return out


def _report_age_days(generated_at: str) -> Optional[float]:
    """报告生成时间距今天数；解析失败返回 None（不猜）。"""
    raw = str(generated_at or "").strip()
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        delta = datetime.now(ts.tzinfo) - ts if ts.tzinfo else datetime.now() - ts
        return delta.total_seconds() / 86400.0
    except Exception:  # noqa: BLE001
        return None


def evaluate_holdout_evidence(holdout: Optional[Dict[str, Any]],
                              config: Optional[Dict[str, Any]] = None,
                              max_report_age_days: Optional[float] = None) -> Dict[str, Any]:
    """把保留期复验报告评估成双指标证据（纯读，不重训、不触网）。

    保留期是**唯一未参与任何扫描**的口径，因此它优先于 walk-forward 曲线。
    """
    cfg = _cfg(config)
    if max_report_age_days is None:
        max_report_age_days = _num(cfg.get("max_report_age_days", 7), 7.0)
    min_hit_rate = _num(cfg.get("min_hit_rate", DEFAULT_MIN_HIT_RATE), DEFAULT_MIN_HIT_RATE)
    min_coverage = _num(cfg.get("min_coverage", DEFAULT_MIN_COVERAGE), DEFAULT_MIN_COVERAGE)
    thr_min = _num(cfg.get("thr_min", DEFAULT_THR_MIN), DEFAULT_THR_MIN)
    thr_max = _num(cfg.get("thr_max", DEFAULT_THR_MAX), DEFAULT_THR_MAX)
    min_samples = int(_num(cfg.get("min_samples", DEFAULT_MIN_SAMPLES), DEFAULT_MIN_SAMPLES))

    generated_at = str((holdout or {}).get("generated_at") or "") if isinstance(holdout, dict) else ""
    age = _report_age_days(generated_at)
    out: Dict[str, Any] = {
        "available": False,
        "report_found": bool(holdout),
        "report_generated_at": generated_at,
        "report_age_days": round(age, 3) if age is not None else None,
        "stale": bool(age is None or age > float(max_report_age_days)),
        "max_report_age_days": float(max_report_age_days),
        "evidence_source": "holdout",
        "single_period_warning": (
            "保留期为单一时段（最近 ~30% 行情），水位不得外推为常态水平；"
            "多时段滚动复验属下一轮工作"
        ),
        "horizons": {},
        "candidate_rank": [],
        "affects_gate": False,
        "reason": "",
    }
    curves = _extract_holdout_curves(holdout)
    if not curves:
        out["reason"] = ("保留期报告缺失或无可用曲线"
                         "（先跑保留期复验：reports/confidence_holdout_verify.json）")
        return out

    any_available = False
    ranking: List[Dict[str, Any]] = []
    for hname, curve in curves.items():
        ev = evaluate_curve(curve, min_hit_rate, min_coverage,
                            thr_min=thr_min, thr_max=thr_max,
                            min_samples=min_samples)
        out["horizons"][hname] = ev
        any_available = any_available or ev["available"]
        if ev["candidate_thresholds"]:
            # 排序用「候选区间内最优命中率」，仅用于呈现，不代选阈值
            best = max((r for r in ev["rows"]
                        if r.get("available") and r.get("in_candidate_range") and r.get("passed")),
                       key=lambda r: (_num(r.get("hit_rate")), _num(r.get("coverage"))),
                       default=None)
            if best:
                ranking.append({
                    "horizon": hname,
                    "threshold": best["threshold"],
                    "hit_rate": best["hit_rate"],
                    "coverage": best["coverage"],
                    "ic": best["ic"],
                    "samples": best["samples"],
                    "note": "呈现用，非推荐阈值；阈值选取属 T15.3 人工签字",
                })
    ranking.sort(key=lambda r: _num(r.get("hit_rate")), reverse=True)
    out["candidate_rank"] = ranking
    out["available"] = any_available
    if out["stale"]:
        out["reason"] = (f"保留期报告过期或时间缺失（generated_at={generated_at or '未知'}，"
                         f"年龄 {out['report_age_days']} 天 > {max_report_age_days} 天）")
    elif not any_available:
        out["reason"] = "保留期曲线全部不可用（样本不足）"
    elif not ranking:
        out["reason"] = (f"保留期各周期在 thr ∈ [{thr_min}, {thr_max}] 内均无双指标同时过线的阈值")
    else:
        out["reason"] = (f"保留期存在候选阈值：{'、'.join(r['horizon'] for r in ranking)}"
                         f"（双指标同时过线，待人工签字）")
    return out


# ----------------------------------------------------------------------
# 决策单（须人工签字）
# ----------------------------------------------------------------------
def build_decision_record(holdout: Optional[Dict[str, Any]] = None,
                          curve: Optional[Dict[str, Any]] = None,
                          config: Optional[Dict[str, Any]] = None,
                          decided_by: str = "",
                          chosen_threshold: Optional[float] = None,
                          reason: str = "",
                          decided_at: Optional[str] = None) -> Dict[str, Any]:
    """生成 T15.3 决策单：提议 + 双指标证据 + 结论 + 人工确认状态。

    **门槛语义**：结论为 `approve` 时，`status` 仍是 `pending`，
    除非 ``decided_by`` 非空（= 有人签字）。没有人工确认的决策单**不得**被
    消费方当作"可改门禁"，这与 `strategy_gate` 的 fail-close 是同一条纪律。
    """
    signed = bool(str(decided_by or "").strip())
    cfg = _cfg(config)
    freeze = bool(cfg.get("freeze_structure", True))

    evidence = evaluate_holdout_evidence(holdout, config)
    # walk-forward 曲线只作补充呈现，不得替代保留期证据
    supplement = None
    if curve is not None:
        min_hit_rate = _num(cfg.get("min_hit_rate", DEFAULT_MIN_HIT_RATE), DEFAULT_MIN_HIT_RATE)
        min_coverage = _num(cfg.get("min_coverage", DEFAULT_MIN_COVERAGE), DEFAULT_MIN_COVERAGE)
        thr_min = _num(cfg.get("thr_min", DEFAULT_THR_MIN), DEFAULT_THR_MIN)
        thr_max = _num(cfg.get("thr_max", DEFAULT_THR_MAX), DEFAULT_THR_MAX)
        min_samples = int(_num(cfg.get("min_samples", DEFAULT_MIN_SAMPLES), DEFAULT_MIN_SAMPLES))
        supplement = {
            "role": "walk_forward_supplement",
            "note": "仅供呈现，非门禁证据（walk-forward 曾参与阈值扫描，有选择自由度）",
            "curve": evaluate_curve(curve, min_hit_rate, min_coverage,
                                    thr_min=thr_min, thr_max=thr_max,
                                    min_samples=min_samples),
        }

    proposed_threshold = None
    if chosen_threshold is not None:
        proposed_threshold = round(_num(chosen_threshold), 4)

    # --- 结论（顺序即优先级，越靠前越"不能改"）---
    verdict = VERDICT_DEFER
    narrative = ""
    if not evidence["report_found"]:
        verdict = VERDICT_DEFER
        narrative = ("缺少保留期复验报告：置信度子集门禁的**唯一独立证据**缺失，"
                     "不能据此改门禁结构。先产保留期报告")
    elif evidence["stale"]:
        verdict = VERDICT_DEFER
        narrative = (f"保留期报告过期或时间缺失（{evidence['reason']}）：证据不可复现，"
                     "先重跑保留期复验再谈改门禁")
    elif not evidence["candidate_rank"]:
        verdict = VERDICT_REJECT
        narrative = (f"保留期上 thr ∈ [{cfg.get('thr_min', DEFAULT_THR_MIN)}, "
                     f"{cfg.get('thr_max', DEFAULT_THR_MAX)}] 内没有阈值同时满足双指标"
                     f"（命中率 ≥ {cfg.get('min_hit_rate', DEFAULT_MIN_HIT_RATE):.2%} 且 "
                     f"覆盖率 ≥ {cfg.get('min_coverage', DEFAULT_MIN_COVERAGE):.2%}）："
                     "不支持按置信度子集重构门禁")
    elif proposed_threshold is None:
        verdict = VERDICT_DEFER
        narrative = (
            "保留期存在双指标同时过线的候选阈值，但**尚未挑定具体阈值**："
            "候选是区间不是单点，需人工显式选定 `chosen_threshold` 后再签"
        )
    elif not any(abs(proposed_threshold - c) < 1e-9
                 for h in evidence["horizons"].values()
                 for c in h.get("candidate_thresholds", [])):
        verdict = VERDICT_REJECT
        narrative = (f"拟选阈值 {proposed_threshold} 不在候选阈值集合内"
                     "（该阈值未同时满足双指标）：不支持")
    elif not signed:
        verdict = VERDICT_DEFER
        narrative = (
            f"候选阈值 {proposed_threshold} 双指标同时过线（保留期口径），证据支持改门禁；"
            "但门禁结构性变更**不能自动生效**：需人工签字（`--decided-by`）"
        )
    else:
        verdict = VERDICT_APPROVE
        narrative = (
            f"候选阈值 {proposed_threshold} 双指标同时过线（保留期口径）且已获人工签字："
            "可作为门禁结构性变更依据。注意——批准的是**证据**，"
            "实际改 `strategy_gate` 仍需人工落配置并重做泄漏/偏差审查，本模块不代改配置"
        )

    status = STATUS_PENDING
    if evidence["stale"]:
        status = STATUS_STALE
    if signed and verdict == VERDICT_APPROVE:
        status = STATUS_CONFIRMED

    blockers: List[str] = []
    if not evidence["report_found"]:
        blockers.append("缺少保留期复验报告")
    if evidence["stale"]:
        blockers.append("保留期报告过期，证据不可复现")
    if not evidence["candidate_rank"]:
        blockers.append("无阈值同时满足双指标")
    if proposed_threshold is None:
        blockers.append("未挑定具体阈值（chosen_threshold 为空）")
    if verdict != VERDICT_APPROVE:
        blockers.append(f"证据结论为 {verdict}：{narrative}")
    if not signed:
        blockers.append("无人工签字（decided_by 为空），门禁结构变更不得自动生效")

    return {
        "generated_at": _now(),
        "decided_at": str(decided_at or ""),
        "decision": {
            "type": "strategy_gate_structure_change",
            "change": ("门禁判据由「全样本命中率」改为"
                       "「置信度 ≥thr 子集命中率 + 覆盖率下限」双指标"),
            "chosen_threshold": proposed_threshold,
            "thr_range": [cfg.get("thr_min", DEFAULT_THR_MIN), cfg.get("thr_max", DEFAULT_THR_MAX)],
            "requires_signature": True,
        },
        "verdict": verdict,
        "status": status,
        "confirmed_by": str(decided_by or ""),
        "reason": str(reason or ""),
        "blockers": blockers,
        "dual_metric": {
            "min_hit_rate": cfg.get("min_hit_rate", DEFAULT_MIN_HIT_RATE),
            "min_coverage": cfg.get("min_coverage", DEFAULT_MIN_COVERAGE),
            "rule": "两条腿必须同时过线；只过一条一律不合格",
        },
        "evidence": evidence,
        "walk_forward_supplement": supplement,
        "freeze_structure": freeze,
        "affects_gate": False,
        "gate_note": (
            "决策单不改变 strategy_gate 结构与放行结论（affects_gate=false）；"
            "即使 status=confirmed，实际改门禁仍须人工落配置并重做泄漏/偏差审查。"
        ),
        "narrative": narrative,
    }


# ----------------------------------------------------------------------
# 落盘 / 读取
# ----------------------------------------------------------------------
def report_path(config: Optional[Dict[str, Any]] = None) -> Path:
    cfg = _cfg(config)
    return Path(cfg.get("report_dir", "reports")) / REPORT_NAME


def holdout_path(config: Optional[Dict[str, Any]] = None) -> Path:
    cfg = _cfg(config)
    return Path(cfg.get("report_dir", "reports")) / DEFAULT_HOLDOUT_NAME


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """读取 JSON；缺失 / 损坏一律返回 None（不猜、不造默认值）。"""
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[confidence-gate] 读取 {path} 失败: {e}")
        return None
