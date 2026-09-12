"""过拟合概率审计：把「试了多少次」变成校正后的显著性（S17 / H2，T17.2 / T17.3）。

问题从哪来：
  S13 把试验登记（append-only）做出来了，S11 / S12 / S15 也把登记次数接进
  了各自的 Bonferroni 校正。但仍是**逐命令**的：`ic` 命令校正时只知道自己
  试过多少次，不知道同一天的另一条命令（不同口径）也在试同一份数据。
  研究者自由度是**跨命令累积**的 —— 这需要一个统一入口。

本模块做什么：
  1. **统一试验预算**（T17.2）：从 append-only 登记里读**全部**命令的累计
     次数，按 `asof` 语义给出"当时已知"的预算，供任意评估命令消费；
     报告里自动标注"本次读数已扫描 N 次"及来源分解（按命令）；
  2. **历史读数回算**（T17.3）：对 S11~S15 已入库的结论（换周期 / 特征扩充 /
     qlib 因子 / 标签 A/B / 置信度门槛）做**统一校正后**的显著性回算，
     输出逐条 `raw p → 校正 p → 是否仍显著`，并如实标注预期（多数维持否定）；
  3. **CPCV 联动**：把 S17 的 CPCV 路径数与收缩指标并入同一份审计报告。

本模块**不做什么**（边界比功能重要）：
  - **不改写历史结论**：只做回算与呈现，既有报告文件一个字节都不动；
  - **不改门禁**：``affects_gate`` 恒为 False；
  - **不猜**：登记文件缺失 = 0 次（尚未登记），损坏 = ``available=false``；
    既有报告缺字段 = 该条标 ``available=false`` + 原因；
  - **不替人工下结论**：回算结果只是材料，是否收紧判据属 T17.4 人工检查点。

无前视说明：
  预算按 ``asof`` 过滤 —— 回算历史读数时只用**当时**已登记的试验次数，
  不把后来的试验算到过去的校正上（否则校正强度被人为放大，结论失真）。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

REPORT_NAME = "overfit_audit.json"

# 历史读数来源：命令 → (报告文件, 说明, 原始 p 的抽取方式)
HISTORY_SOURCES: Sequence[Dict[str, str]] = (
    {"stage": "S11", "topic": "预测周期切换（多重比较校正）",
     "file": "reports/horizon_decision.json", "command": "horizon-decision"},
    {"stage": "S12", "topic": "特征扩充正交对照",
     "file": "reports/feature_experiment.json", "command": "feature-experiment"},
    {"stage": "S13", "topic": "qlib Alpha158 因子增量",
     "file": "reports/qlib_ab.json", "command": "qlib-ab"},
    {"stage": "S12", "topic": "三重障碍法标签 A/B",
     "file": "reports/label_ab.json", "command": "label-ab"},
    {"stage": "S15", "topic": "置信度子集门禁（双指标）",
     "file": "reports/confidence_gate_decision.json", "command": "confidence-gate"},
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[overfit-audit] {path} 无法解析: {e}")
        return None
    return data if isinstance(data, dict) else None


def effective_trial_budget(config: Optional[Dict[str, Any]],
                           asof: Optional[str] = None,
                           commands: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """统一试验预算（T17.2）：跨命令累计次数 + 来源分解。

    返回 ``{"available", "count", "total", "by_command", "asof", "reason"}``。
    登记不可用（文件损坏）→ ``available=false`` 且 ``count`` 置 0 **并给原因**
    —— 调用方必须据此标注"预算未知"，不得静默当作 1 次。
    """
    try:
        from src.eval.trial_registry import count_trials

        stats = count_trials(config, asof=asof)
    except Exception as e:  # noqa: BLE001
        return {"available": False, "count": 0, "total": 0, "by_command": {},
                "asof": str(asof or ""), "reason": str(e)}
    if not stats.get("available"):
        return {"available": False, "count": 0, "total": 0, "by_command": {},
                "asof": str(asof or ""),
                "reason": stats.get("reason", "trial registry unavailable")}

    by_command = dict(stats.get("by_command") or {})
    if commands:
        wanted = {str(c) for c in commands}
        by_command = {k: v for k, v in by_command.items() if k in wanted}
    count = int(sum(int(v) for v in by_command.values())) if commands else int(
        stats.get("count", 0))
    return {
        "available": True,
        "count": count,
        "total": int(stats.get("total", 0)),
        "by_command": by_command,
        "asof": str(asof or ""),
        "file": stats.get("file", ""),
        "reason": "",
    }


def annotate_selection_freedom(budget: Dict[str, Any], n_this_run: int = 1) -> Dict[str, Any]:
    """把试验预算变成报告里可引用的一段标注（"本次读数已扫描 N 次"）。"""
    n_this = max(1, int(n_this_run))
    if not budget.get("available"):
        return {
            "available": False,
            "n_trials_effective": n_this,
            "note": (f"本次读数已扫描 {n_this} 次；历史登记不可用"
                     f"（{budget.get('reason', '未知原因')}），校正强度可能被低估"),
        }
    hist = int(budget.get("count", 0))
    total = hist + n_this
    return {
        "available": True,
        "n_trials_history": hist,
        "n_trials_this_run": n_this,
        "n_trials_effective": total,
        "by_command": budget.get("by_command", {}),
        "note": (f"本次读数已扫描 {total} 次"
                 f"（历史 {hist} 次 + 本次 {n_this} 次），多重比较校正应按 {total} 计"),
    }


def _z_to_p(z: Optional[float]) -> Optional[float]:
    """z 值 → 双侧 p 值（正态近似，与 feature_experiment 同源口径）。"""
    if z is None:
        return None
    try:
        from src.eval.horizon_decision import normal_two_sided_p

        return float(normal_two_sided_p(float(z)))
    except Exception:  # noqa: BLE001
        import math

        try:
            return float(math.erfc(abs(float(z)) / math.sqrt(2.0)))
        except Exception:  # noqa: BLE001
            return None


def _extract_raw_p(report: Optional[Dict[str, Any]]) -> Optional[float]:
    """从既有报告里抽"最有利的原始 p"（兼容多种布局）。抽不到 → None（不猜）。

    抽取顺序（先精确字段、后统计量换算）：
      1. 任何形如 ``p_value`` / ``p_family`` / ``p`` 的数值字段（取最小值）；
      2. 没有 p 字段时，从 ``z`` / ``hit_rate_z_*`` 等 z 统计量换算出 p
         （两步法：先试着收集 p，拿不到再换算 z —— 不在同一层混用，避免
         把"换算出来的 p"伪装成"报告里原本就有的 p"）。
    """
    if not report:
        return None

    p_cands: List[float] = []
    z_cands: List[float] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                key = str(k).lower()
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    if key in ("p", "p_value", "p_raw", "p_family") or "p_value" in key:
                        p_cands.append(float(v))
                    elif (key == "z" or key.endswith("_z") or key.startswith("z_")
                          or "_z_" in key or key.endswith("_z_new") or key.endswith("_z_old")):
                        z_cands.append(float(v))
                _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(report)
    p_ok = [c for c in p_cands if 0.0 <= c <= 1.0]
    if p_ok:
        return float(min(p_ok))
    z_ok = [c for c in z_cands if abs(c) < 1e6]
    if z_ok:
        # 最有利 = |z| 最大（z 越大越显著）
        zs = max(z_ok, key=lambda x: abs(x))
        return _z_to_p(zs)
    return None


def recompute_history(config: Optional[Dict[str, Any]] = None,
                      reports_dir: str = "reports",
                      sources: Sequence[Dict[str, str]] = HISTORY_SOURCES,
                      asof: Optional[str] = None) -> List[Dict[str, Any]]:
    """对历史读数逐条回算校正后显著性（T17.3）。

    每条读数：
      - 原始 p（报告里最有利的那个）；
      - 该命令当时的累计试验次数（``asof`` 口径）；
      - Bonferroni 校正后 p 与其是否仍显著（判据 0.05）；
      - 结果不显著时如实写"校正后不显著"，**不修饰**。
    """
    from src.eval.horizon_decision import bonferroni

    d = Path(reports_dir)
    rows: List[Dict[str, Any]] = []
    for src in sources:
        report = _load_json(d / Path(src["file"]).name)
        raw_p = _extract_raw_p(report)
        budget = effective_trial_budget(config, asof=asof, commands=[src["command"]])
        n_trials = max(1, int(budget.get("count", 0))) if budget.get("available") else None
        row: Dict[str, Any] = {
            "stage": src["stage"],
            "topic": src["topic"],
            "source_file": src["file"],
            "command": src["command"],
            "report_found": report is not None,
            "raw_min_p": None if raw_p is None else round(raw_p, 6),
            "n_trials_asof": n_trials,
            "trial_budget_available": bool(budget.get("available")),
            "n_trials_note": budget.get("reason", ""),
            "p_adjusted": None,
            "still_significant": None,
            "verdict_note": "",
        }
        if report is None or raw_p is None:
            row["verdict_note"] = ("报告缺失" if report is None
                                   else "报告中未找到原始 p 值（不猜）")
        elif n_trials is None:
            row["verdict_note"] = "试验预算不可用，无法校正（不猜）"
        else:
            adj = bonferroni(raw_p, n_trials)
            row["p_adjusted"] = round(adj, 6)
            row["still_significant"] = bool(adj <= 0.05)
            row["verdict_note"] = (
                f"按 {n_trials} 次试验 Bonferroni 校正：p {raw_p:.4f} → {adj:.4f}，"
                + ("仍显著" if row["still_significant"] else "校正后**不显著**")
            )
        rows.append(row)
    return rows


def build_report(config: Optional[Dict[str, Any]] = None,
                 reports_dir: str = "reports",
                 n_this_run: int = 1,
                 cpcv_summary: Optional[Dict[str, Any]] = None,
                 asof: Optional[str] = None) -> Dict[str, Any]:
    """构造过拟合审计报告（T17.2 + T17.3 + CPCV 联动）。"""
    budget = effective_trial_budget(config, asof=asof)
    freedom = annotate_selection_freedom(budget, n_this_run=n_this_run)
    history = recompute_history(config, reports_dir=reports_dir, asof=asof)

    usable = [r for r in history if r["still_significant"] is not None]
    still_sig = [r for r in usable if r["still_significant"]]
    return {
        "kind": "overfit_audit",
        "generated_at": _now(),
        "asof": str(asof or ""),
        "trial_budget": budget,
        "selection_freedom": freedom,
        "cpcv": cpcv_summary or {"available": False, "reason": "未提供 CPCV 摘要"},
        "history_recompute": history,
        "summary": {
            "n_history_readings": len(history),
            "n_recomputed": len(usable),
            "n_still_significant": len(still_sig),
            "note": ("回算只做呈现，不改写任何既有报告；多数历史结论预期维持否定"
                     "（S11 换周期 / S12 特征扩充 / S13 因子增量 / S15 门槛）"),
        },
        "affects_gate": False,
        "note": ("是否引入过拟合概率下限属 T17.4 人工检查点，本报告不代改判据。"),
    }


def save(report: Dict[str, Any], reports_dir: str = "reports",
         name: str = REPORT_NAME) -> Path:
    d = Path(reports_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / name
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
