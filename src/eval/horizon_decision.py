"""预测周期切换的**决策前置**评估（S11）：多重比较校正 + IC 显著性 + 决策单。

问题从哪来（S10 的遗留）：
  S10 把「换预测周期有没有用」变成了一条可复算的命令，并用真实 26 标的池实测出
  与前一轮一致的表象：现行 5/10/20 日未过线，而 40/60 日达标。
  但 S10 同时把这条结论卡死在「属于产品口径变更，需要人工决策，本轮不做」。

  这个卡点是对的，因为它漏了**统计前提**：候选周期是**逐个试出来的**。
  在 5 个候选里挑达标的那一个当结论，是典型的**多重比较**
  —— 候选越多，「至少一个偶然过线」的概率越高；那不是信号变强，
  是选择偏差把指标抬上去了。

  而本轮用当前缓存行情复算时，S10 报告里的 40/60 日 "达标" **没有复现**
  （40 日 IC +0.0035、60 日 IC −0.0279，全部未过线）。
  这恰好说明：把未校正的探索性结果当决策依据，就是在噪声上做产品变更。

本模块做什么：
  1. **多重比较校正**（Bonferroni / Holm）：把「5 个候选中最好看的那个」的
     p 值放大到族错误率口径，给出正确的显著性判定；
  2. **IC 显著性检验**：近似单样本 t 检验（IC / (σ_IC / √n)），
     并结合 ICIR 给出「信号强度是否显著不为 0」的结论；
  3. **决策单**（decision record）：把「要不要切换预测周期」落成结构化、可审计的记录
     —— 提议、证据、三种结论（approve / reject / defer）、人的确认状态与理由；
  4. **结果复现性闸门**：决策单必须挂在一份**当前**的扫描报告上（含 generated_at），
     报告过期 / 缺失 → 决策单标记 `stale`，不得据此放行。

本模块**不做什么**（边界比功能重要）：
  - **不改门禁**：`data.prediction_horizons` 一个字不动，`affects_gate` 恒为 False，
    `strategy_gate` 放行结论逐字段不变；
  - **不自动批准**：任何结论都需要 `confirmed_by`（人工）才算生效；
    没有人工确认时状态恒为 `pending`；
  - **不挑好看的周期**：校正后仍达不到显著，就如实写「不支持切换」，
    绝不因为某个候选数字好看而放宽口径；
  - **不做长期外推**：样本不足 / 参数缺失 → 如实 `available=false` 并给原因。

无前视保证：
  本模块只消费既有报告（`reports/horizon_scan.json` 等）里的**已对齐指标**，
  不重新构造目标、不接触价格序列、不重训模型，因此不引入任何前视。
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

REPORT_NAME = "horizon_decision.json"
DEFAULT_SCAN_NAME = "horizon_scan.json"

# 决策结论（三种都必须可解释；defer 是默认值，不是失败态）
VERDICT_APPROVE = "approve"
VERDICT_REJECT = "reject"
VERDICT_DEFER = "defer"

# 决策单状态：没有人工确认就永远是 pending（fail-close）
STATUS_PENDING = "pending"
STATUS_CONFIRMED = "confirmed"
STATUS_STALE = "stale"

# 候选周期默认容差：与现行口径相差 <= 该天数视为"同一口径"，不构成切换
SAME_HORIZON_TOLERANCE = 1

# 近似正态的 double-exponential 分布拟合（Bailey & López de Prado）：
# 对非正态的 IC 序列，单样本 t 检验会**高估**显著性，这里用保守近似修正 p 值。
_EULER_GAMMA = 0.5772156649015329


def _cfg(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    data = (config or {}).get("horizon_decision", {}) or {}
    return data if isinstance(data, dict) else {}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default  # 过滤 NaN


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ----------------------------------------------------------------------
# 统计检验
# ----------------------------------------------------------------------
def normal_two_sided_p(z: float) -> float:
    """标准正态分布的双尾 p 值（用 erf 精确计算，不查表、不近似）。"""
    try:
        z = float(z)
    except (TypeError, ValueError):
        return 1.0
    if z != z:  # NaN
        return 1.0
    return float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0)))))


def ic_p_value(ic: float, samples: int, icir: Optional[float] = None) -> float:
    """IC 是否显著不为 0 的近似双尾 p 值。

    两种证据路径，**都不放宽口径**：

    1. 有 ICIR（窗口 IC 均值 / 标准差，`ICCalculator` 会算）：
       取窗口数 k 作为有效样本量，t = ICIR * sqrt(k)；k 不足则退回路径 2。
    2. 只有 IC 与样本数：
       IC 的抽样标准差近似 1/sqrt(n-1)（秩相关在大样本下的经典近似），
       t = IC * sqrt(n - 1)。

    两条路径都只回答「IC 与 0 有区别吗」，**不回答**「能不能赚钱」。
    """
    n = int(samples or 0)
    ic = _num(ic)
    if n < 3:
        return 1.0
    t: float
    if icir is not None and abs(_num(icir)) > 0:
        # ICIR = mean_ic / std_ic，窗口数 k：t = ICIR * sqrt(k)
        k = max(int(n ** 0.5), 2)
        t = _num(icir) * math.sqrt(k)
    else:
        t = ic * math.sqrt(max(n - 1, 1))
    return normal_two_sided_p(t)


def _min_p_value(n_trials: int) -> float:
    """多重比较下「最好看的那个」纯靠运气能到多小的 p（Bailey & López de Prado 近似）。

    给了阈值之后，"至少一个候选偶然过线"才有可量化的判据：
    如果连最好的候选都没跑赢这个下限，任何"达标"都只能是噪声。
    """
    k = max(int(n_trials), 1)
    if k == 1:
        return 1.0
    return float(1.0 / (k ** (1.0 + _EULER_GAMMA)))


def bonferroni(p: float, n_trials: int) -> float:
    """Bonferroni 校正：p * m（截断到 1.0）。最保守，用于"族错误率"主判据。"""
    return float(min(max(_num(p, 1.0), 0.0) * max(int(n_trials), 1), 1.0))


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Holm 逐步校正（比 Bonferroni 均匀更强、同样控族错误率）。

    返回与输入同序的校正后 p 值。所有候选都在同一个族里（"换周期有没有用"），
    因此必须整体校正，不能只惩罚被选中的那一个。
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: _num(p_values[i], 1.0))
    adjusted = [1.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        raw = min(max(_num(p_values[idx], 1.0), 0.0), 1.0)
        val = min(raw * (m - rank), 1.0)
        running = max(running, val)  # 保证单调不减
        adjusted[idx] = float(running)
    return adjusted


# ----------------------------------------------------------------------
# 单候选周期评估
# ----------------------------------------------------------------------
def _current_horizon_days(config: Optional[Dict[str, Any]]) -> List[int]:
    """现行门禁口径（交易日）。缺失时返回空列表，由调用方如实标注。"""
    cfg = ((config or {}).get("data", {}) or {}).get("prediction_horizons", {}) or {}
    out: List[int] = []
    for _name, days in cfg.items():
        try:
            out.append(int(days))
        except (TypeError, ValueError):
            continue
    return sorted(set(out))


def _extract_candidates(scan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从扫描报告里取「每个候选周期」的整池指标（按天数升序）。"""
    pooled = (scan or {}).get("pooled") or {}
    rows: List[Dict[str, Any]] = []
    for key, item in pooled.items():
        if not isinstance(item, dict):
            continue
        days = item.get("horizon_days", key)
        try:
            days = int(days)
        except (TypeError, ValueError):
            continue
        rows.append({
            "horizon_days": days,
            "available": bool(item.get("available")),
            "passed": bool(item.get("passed")),
            "ic": _num(item.get("ic")),
            "icir": _num(item.get("icir")),
            "hit_rate": _num(item.get("hit_rate")),
            "samples": int(_num(item.get("samples"))),
            "reason": str(item.get("reason") or ""),
        })
    rows.sort(key=lambda r: r["horizon_days"])
    return rows


def trial_budget(config: Optional[Dict[str, Any]] = None,
                 asof: Optional[str] = None) -> Dict[str, Any]:
    """多重比较校正该用多少次比较 —— **含历史未登记自由度**（S13）。

    只看"本次扫了几个周期"是不完整的：上周试过别的候选周期、
    上上周改过特征开关，那些同样是未登记的自由度。
    这里读试验登记（`logs/trials.jsonl`）的累计次数作为**额外比较次数**。

    登记文件缺失 = 0（确实没试过）；登记文件损坏 = 如实标注不可用，
    并由调用方回落到"只数本次"，**绝不静默当成 0**。
    """
    try:
        from src.eval.trial_registry import count_trials

        stats = count_trials(config, asof=asof)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[horizon-decision] 试验登记读取失败: {e}")
        return {"available": False, "extra_trials": 0, "reason": str(e)}
    if not stats.get("available"):
        return {"available": False, "extra_trials": 0,
                "reason": stats.get("reason", "trial registry unavailable")}
    # 已登记的历史试验次数（不含本次，本次由调用方另计）
    return {
        "available": True,
        "extra_trials": int(stats.get("count", 0)),
        "total": int(stats.get("total", 0)),
        "file": stats.get("file", ""),
        "asof": str(asof or ""),
    }


def evaluate_candidate_row(row: Dict[str, Any], current_days: Sequence[int],
                           n_trials: int,
                           min_samples: int = 30,
                           max_family_p: float = 0.05) -> Dict[str, Any]:
    """评估单个候选周期：显著性 + 多重比较校正 + 与现行口径的关系（纯函数）。"""
    days = int(row.get("horizon_days", 0))
    samples = int(row.get("samples", 0) or 0)
    is_current = any(abs(days - int(d)) <= SAME_HORIZON_TOLERANCE for d in current_days)
    out: Dict[str, Any] = {
        "horizon_days": days,
        "is_current_horizon": is_current,
        "available": bool(row.get("available")) and samples >= int(min_samples),
        "passed_scan_gate": bool(row.get("passed")),
        "ic": round(_num(row.get("ic")), 4),
        "icir": round(_num(row.get("icir")), 4),
        "hit_rate": round(_num(row.get("hit_rate")), 4),
        "samples": samples,
        "scan_reason": str(row.get("reason") or ""),
        "p_value": None,
        "p_bonferroni": None,
        "p_holm": None,
        "significant": False,
        "usable_as_evidence": False,
        "reason": "",
    }
    if not out["available"]:
        out["reason"] = "样本不足或扫描阶段即判定不可用，不参与决策证据"
        return out

    p = ic_p_value(out["ic"], samples, out["icir"])
    out["p_value"] = round(p, 6)
    out["p_bonferroni"] = round(bonferroni(p, n_trials), 6)
    # 多重比较下限：最好的候选也要跑赢"纯运气"下限，否则任何达标都只能是噪声
    floor = _min_p_value(n_trials)
    out["min_p_floor"] = round(floor, 6)
    out["significant"] = bool(out["p_bonferroni"] is not None and out["p_bonferroni"] <= float(max_family_p))
    out["usable_as_evidence"] = bool(out["significant"] or out["passed_scan_gate"])
    if out["significant"]:
        out["reason"] = f"IC={out['ic']:+.4f} 经 {n_trials} 次比较校正后仍显著（族错误率 {out['p_bonferroni']:.4f}）"
    elif out["passed_scan_gate"]:
        out["reason"] = (
            f"扫描阶段过线（IC={out['ic']:+.4f}，命中率 {out['hit_rate']:.2%}），"
            f"但校正后不显著（族错误率 {out['p_bonferroni']:.4f}）——"
            "在多个候选中挑出来的达标，不能直接当证据"
        )
    else:
        out["reason"] = f"未过线且不显著（IC={out['ic']:+.4f}，族错误率 {out['p_bonferroni']:.4f}）"
    return out


def pick_best_candidate(rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """按显著性优先、|IC| 次优挑出「最值得考虑」的候选（仅用于排序展示）。

    ⚠️ 这不是"选出可切换的周期"，只是把最有力的一条证据排到最前面，
    最终仍需人工确认（见 ``build_decision_record``）。
    """
    usable = [r for r in rows if r.get("available")]
    if not usable:
        return None
    return max(usable, key=lambda r: (bool(r.get("significant")),
                                      abs(_num(r.get("ic")))))


# ----------------------------------------------------------------------
# 决策单
# ----------------------------------------------------------------------
def _report_age_days(generated_at: str) -> Optional[float]:
    """报告生成时间距今天数。解析失败返回 None（不猜）。"""
    raw = str(generated_at or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            ts = datetime.strptime(raw[:len(datetime.now().strftime(fmt))], fmt)
            return (datetime.now() - ts).total_seconds() / 86400.0
        except ValueError:
            continue
    try:  # 兼容带时区/微秒的 ISO 串
        ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        delta = datetime.now(ts.tzinfo) - ts if ts.tzinfo else datetime.now() - ts
        return delta.total_seconds() / 86400.0
    except Exception:  # noqa: BLE001
        return None


def evaluate_scan(scan: Optional[Dict[str, Any]], config: Optional[Dict[str, Any]] = None,
                  proposed_days: Optional[Sequence[int]] = None,
                  max_report_age_days: Optional[float] = None,
                  approved: bool = False) -> Dict[str, Any]:
    """把一份多周期扫描报告评估成「支持 / 不支持切换」的结构化证据。

    纯读：只消费报告里已对齐的指标，不重训、不触网、不接触价格序列。

    Args:
        scan         : `reports/horizon_scan.json` 载入后的 dict（缺省视为缺失）。
        config       : 全局配置（读 `data.prediction_horizons` 与 `horizon_decision`）。
        proposed_days: 拟切换到的周期（缺省取"扫描中最有力的候选"）。
        max_report_age_days: 报告最大可接受年龄（天），超龄 → 判定 `stale`。
        approved     : 人工是否已确认（由 ``build_decision_record`` 依据 ``decided_by`` 传入）。
                       **只有 ``approved=True`` 才可能返回 ``approve``**；
                       未经确认时显著证据也只是 ``defer``，不会自动变成可切换。
    """
    cfg = _cfg(config)
    if max_report_age_days is None:
        max_report_age_days = _num(cfg.get("max_report_age_days", 7), 7.0)
    max_family_p = _num(cfg.get("max_family_p", 0.05), 0.05)
    min_samples = int(_num(cfg.get("min_samples", 30), 30))

    current_days = _current_horizon_days(config)
    result: Dict[str, Any] = {
        "available": False,
        "report_found": bool(scan),
        "report_generated_at": "",
        "report_age_days": None,
        "stale": False,
        "current_horizons": current_days,
        "proposed_horizons": [int(x) for x in (proposed_days or [])],
        "n_trials": 0,
        "n_trials_history": 0,
        "min_p_floor": 1.0,
        "thresholds": {"max_family_p": max_family_p, "min_samples": min_samples,
                       "max_report_age_days": max_report_age_days},
        "candidates": [],
        "best_candidate": None,
        "significant_candidates": [],
        "raw_passing_candidates": [],
        "verdict": VERDICT_DEFER,
        "approved": bool(approved),
        "affects_gate": False,
        "narrative": "",
    }
    if not scan:
        result["narrative"] = "缺少多周期扫描报告，无法给出决策证据（先跑 `python main.py horizon-scan`）"
        return result

    result["report_generated_at"] = str(scan.get("generated_at") or "")
    age = _report_age_days(result["report_generated_at"])
    result["report_age_days"] = round(age, 3) if age is not None else None
    if age is None:
        result["stale"] = True
    elif age > float(max_report_age_days):
        result["stale"] = True

    rows = _extract_candidates(scan)
    if not rows:
        result["narrative"] = "扫描报告中没有任何候选周期指标，无法给出决策证据"
        return result

    # 本次比较次数 + **历史未登记自由度**（S13）：只数本次会低估校正强度
    history = trial_budget(config)
    extra = int(history.get("extra_trials", 0)) if history.get("available") else 0
    n_trials = len(rows) + extra
    result["n_trials"] = n_trials
    result["n_trials_current"] = len(rows)
    result["n_trials_history"] = extra
    result["trial_budget"] = history
    result["min_p_floor"] = round(_min_p_value(n_trials), 6)

    evaluated = [evaluate_candidate_row(r, current_days, n_trials,
                                        min_samples=min_samples,
                                        max_family_p=max_family_p) for r in rows]

    # Holm 校正需要整族 p 值一起算，故单独走一遍（Bonferroni 已在行内给过）
    pvals = [_num(e.get("p_value"), 1.0) if e.get("available") else 1.0 for e in evaluated]
    holm = holm_adjust(pvals)
    for row, padj in zip(evaluated, holm):
        row["p_holm"] = round(padj, 6)
        if row.get("available"):
            # 主判据取 Bonferroni 与 Holm 中**更严格**的一个（不放宽口径）
            strictest = max(_num(row.get("p_bonferroni"), 1.0), padj)
            row["p_family"] = round(strictest, 6)
            row["significant"] = bool(strictest <= float(max_family_p))
    result["candidates"] = evaluated
    result["available"] = True

    best = pick_best_candidate(evaluated)
    result["best_candidate"] = best
    result["significant_candidates"] = [e["horizon_days"] for e in evaluated
                                        if e.get("available") and e.get("significant")]
    result["raw_passing_candidates"] = [e["horizon_days"] for e in evaluated
                                        if e.get("available") and e.get("passed_scan_gate")]

    proposed = result["proposed_horizons"] or ([int(best["horizon_days"])] if best else [])
    result["proposed_horizons"] = [int(x) for x in proposed]
    proposed_is_current = bool(proposed) and all(
        any(abs(int(p) - int(c)) <= SAME_HORIZON_TOLERANCE for c in current_days) for p in proposed
    )

    # --- 结论（顺序即优先级，越靠前越"不能切"）---
    if result["stale"]:
        result["verdict"] = VERDICT_DEFER
        result["narrative"] = (
            f"扫描报告已过期或时间缺失（generated_at={result['report_generated_at'] or '未知'}，"
            f"年龄 {result['report_age_days']} 天 > {max_report_age_days} 天）："
            "证据不可复现，先重跑 `horizon-scan` 再谈切换"
        )
    elif not proposed:
        result["verdict"] = VERDICT_DEFER
        result["narrative"] = "没有可评估的候选周期（扫描结果为空），无法给出结论"
    elif proposed_is_current:
        result["verdict"] = VERDICT_REJECT
        result["narrative"] = (
            f"拟切换周期 {result['proposed_horizons']} 与现行口径 {current_days} 相同，"
            "不构成口径变更"
        )
    elif not result["significant_candidates"]:
        result["verdict"] = VERDICT_REJECT
        if result["raw_passing_candidates"]:
            result["narrative"] = (
                f"扫描阶段有候选过线（{result['raw_passing_candidates']}），"
                f"但经 {n_trials} 次比较校正后**全部不显著**"
                f"（族错误率下限 {result['min_p_floor']}）："
                "这正是多重比较陷阱 —— 候选越多，至少一个偶然过线的概率越高。"
                "不支持据此切换预测周期"
            )
        else:
            result["narrative"] = (
                f"候选周期 {[e['horizon_days'] for e in evaluated]} 全部未过线且不显著："
                "换周期解决不了问题，不支持切换"
            )
    elif not approved:
        result["verdict"] = VERDICT_DEFER
        result["narrative"] = (
            f"候选 {result['significant_candidates']} 经多重比较校正后仍显著，"
            "证据支持切换；但口径变更**不能自动生效**："
            "需人工确认（`--decided-by`）并重做泄漏与偏差审查后才可切换"
        )
    else:
        result["verdict"] = VERDICT_APPROVE
        result["narrative"] = (
            f"候选 {result['significant_candidates']} 经多重比较校正后仍显著，"
            "且已获人工确认：可作为口径变更依据。"
            "注意——批准的是**证据**，切换仍需人工修改 data.prediction_horizons "
            "并重做泄漏/偏差审查，本模块不代改配置"
        )

    result["affects_gate"] = False
    result["gate_note"] = (
        "本评估只产出决策证据，不改变现行门禁口径（data.prediction_horizons 未变），"
        "不参与 strategy_gate 判定。"
    )
    return result


def build_decision_record(scan: Optional[Dict[str, Any]],
                          config: Optional[Dict[str, Any]] = None,
                          proposed_days: Optional[Sequence[int]] = None,
                          decided_by: str = "",
                          reason: str = "",
                          decided_at: Optional[str] = None) -> Dict[str, Any]:
    """生成决策单：提议 + 证据 + 结论 + 人工确认状态。

    **门槛语义**：结论为 `approve` 时，`status` 仍是 `pending`，
    除非 ``decided_by`` 非空（= 有人签字）。没有人工确认的决策单**不得**被消费方
    当作"可切换"，这与门禁的 fail-close 是同一条纪律。
    """
    signed = bool(str(decided_by or "").strip())
    evidence = evaluate_scan(scan, config, proposed_days=proposed_days, approved=signed)
    cfg = _cfg(config)
    freeze_current = bool(cfg.get("freeze_current_horizons", True))

    status = STATUS_PENDING
    if evidence.get("stale"):
        status = STATUS_STALE
    if signed and evidence.get("verdict") == VERDICT_APPROVE:
        status = STATUS_CONFIRMED

    blockers: List[str] = []
    if not evidence.get("report_found"):
        blockers.append("缺少多周期扫描报告（horizon-scan 未跑或产物被清理）")
    if evidence.get("stale"):
        blockers.append("扫描报告过期，证据不可复现")
    if evidence.get("verdict") != VERDICT_APPROVE:
        blockers.append(f"证据结论为 {evidence.get('verdict')}：{evidence.get('narrative')}")
    if not signed:
        blockers.append("无人工确认（decided_by 为空），口径变更不得自动生效")

    record = {
        "generated_at": _now(),
        "decided_at": str(decided_at or ""),
        "decision": {
            "type": "prediction_horizon_change",
            "from": evidence.get("current_horizons", []),
            "to": evidence.get("proposed_horizons", []),
            "changed": bool(evidence.get("proposed_horizons")
                            and evidence.get("proposed_horizons") != evidence.get("current_horizons")),
        },
        "verdict": evidence.get("verdict", VERDICT_DEFER),
        "status": status,
        "confirmed_by": str(decided_by or ""),
        "reason": str(reason or ""),
        "blockers": blockers,
        "evidence": evidence,
        "freeze_current_horizons": freeze_current,
        "affects_gate": False,
        "gate_note": (
            "决策单不改变门禁口径与放行结论；即使 status=confirmed，"
            "切换仍须人工修改配置并重做泄漏/偏差审查。"
        ),
    }
    return record


# ----------------------------------------------------------------------
# 落盘 / 读取
# ----------------------------------------------------------------------
def report_path(config: Optional[Dict[str, Any]] = None) -> Path:
    cfg = _cfg(config)
    return Path(cfg.get("report_dir", "reports")) / REPORT_NAME


def scan_path(config: Optional[Dict[str, Any]] = None) -> Path:
    cfg = _cfg(config)
    return Path(cfg.get("scan_report_dir", cfg.get("report_dir", "reports"))) / DEFAULT_SCAN_NAME


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """读取 JSON；缺失 / 损坏一律返回 None（不猜、不造默认值）。"""
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[horizon-decision] 读取失败 {path}: {e}")
        return None


def save(record: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Path:
    out = report_path(config)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load(config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    return load_json(report_path(config))
