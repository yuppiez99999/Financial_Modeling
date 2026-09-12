"""投研辅助链路（S20 / H5，T20.1~T20.4）：LLM 投研结论**只读**接入。

问题从哪来（为什么"最后才做、还可能不做"）：
  本项目门禁一直卡在命中率线上，而 S16~S19 已经把「概率语义 / 评估可信度 /
  条件有效性」加固完。投研辅助（基本面、事件、情绪的中文研报式解读）是
  **另一个信息维度**，但它有两个硬风险：
    1. 依赖重（LLM 调用要网络 + 成本 + 不可复现）；
    2. 噪声大（LLM 结论无法量化时，接进信号路径等于给门禁注入随机性）。

所以 H5 的纪律比功能重要得多（排期里就已写明）：

  > **只读、不进信号路径、评估不通过则整阶段取消。**

本模块做什么（严格按这三条落地）：
  1. **调研评估**（T20.1）：把候选项目（TradingAgents / TradingAgents-CN /
     QuantMind）的适配情况落成一页评估 —— 是否值得引入、依赖代价、许可证。
     结论可以是"不引入"，且**不引入也是有效交付**；
  2. **只读附注**（T20.2）：LLM 结论只作报告附注（``research_note``），
     类型上就与信号/概率/门禁**不连通**（本模块不产出任何可进入判据的字段）；
  3. **离线对照**（T20.3）：同一批样本上「有/无附注」的命中率差异 ——
     但附注不影响预测，所以差异**在结构上必然为 0**；无法量化时如实记录为
     ``unverifiable`` 并**止损**，不编造"LLM 提升命中率"的故事；
  4. **保留决策**（T20.4）：决策单，默认 ``defer`` / ``cancel``，无人工签字不放行。

本模块**不做什么**（边界比功能重要）：
  - **不进信号路径**：``affects_signal`` / ``affects_gate`` 恒为 False ——
    这是结构性保证，不是"约定"；
  - **不调用真实 LLM**：离线优先，附注由调用方注入或留空；
  - **不编造效果**：无法量化就写 ``unverifiable`` + 止损理由；
  - **不签字**：无 ``decided_by`` → 结论最多 ``defer`` / ``cancel``。

无前视说明：
  附注只挂在**报告层**，不参与任何特征构造与打分；对照实验用的是已有预测
  与真实收益，不引入未来信息。
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# 候选项目（S20 / H5 排期给定）
CANDIDATES: Sequence[Dict[str, Any]] = (
    {
        "repo": "TauricResearch/TradingAgents",
        "license": "Apache-2.0",
        "stars_hint": "104k",
        "a_share_adaptation": "无（美股/通用）",
        "dependency_cost": "高：LLM API + 多智能体编排 + 网络",
        "offline_reproducible": False,
        "verdict": "reject",
        "reason": "原版无 A 股数据适配；依赖 LLM API 与网络，违背『CI 离线可跑』纪律",
    },
    {
        "repo": "hsliuping/TradingAgents-CN",
        "license": "Apache-2.0 系（需逐项核）",
        "stars_hint": "31.7k",
        "a_share_adaptation": "有（A 股数据源适配 + 中文研报输出）",
        "dependency_cost": "中高：LLM API（可换本地模型）+ 多智能体 + 网络",
        "offline_reproducible": False,
        "verdict": "defer",
        "reason": ("A 股适配最好，但核心价值依赖 LLM 调用；本项目离线 CI 无法复现，"
                   "只适合作为**人工投研辅助工具**，不宜进仓库链路"),
    },
    {
        "repo": "qusong0627/QuantMind",
        "license": "未标注（NOASSERTION）",
        "stars_hint": "1.4k",
        "a_share_adaptation": "部分",
        "dependency_cost": "中：LLM + 数据抓取",
        "offline_reproducible": False,
        "verdict": "reject",
        "reason": "许可证未明确（NOASSERTION），引入存在合规风险；收益不明确",
    },
)

STATUS_PENDING = "pending"
STATUS_CONFIRMED = "confirmed"
VERDICT_DEFER = "defer"
VERDICT_CANCEL = "cancel"
VERDICT_PROCEED = "proceed"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_research_evaluation(candidates: Sequence[Dict[str, Any]] = CANDIDATES
                              ) -> Dict[str, Any]:
    """一页评估（T20.1）：逐候选给"是否值得引入 / 依赖代价 / 许可证"。

    ``worth_introducing`` 只有在该候选同时满足「有 A 股适配 + 离线可复现 +
    许可证明确」时才为真 —— 任一不满足即 False，并写明卡在哪一条。
    """
    rows: List[Dict[str, Any]] = []
    for c in candidates:
        license_ok = "NOASSERTION" not in str(c.get("license", "")).upper()
        offline_ok = bool(c.get("offline_reproducible"))
        a_share_ok = "无" not in str(c.get("a_share_adaptation", ""))
        blockers: List[str] = []
        if not license_ok:
            blockers.append("许可证不明确")
        if not offline_ok:
            blockers.append("离线不可复现（依赖 LLM 调用/网络）")
        if not a_share_ok:
            blockers.append("无 A 股适配")
        row = dict(c)
        row["license_clear"] = license_ok
        row["a_share_adapted"] = a_share_ok
        row["worth_introducing"] = bool(license_ok and offline_ok and a_share_ok)
        row["blockers"] = blockers
        rows.append(row)

    introducible = [r for r in rows if r["worth_introducing"]]
    return {
        "kind": "research_assist_evaluation",
        "generated_at": _now(),
        "candidates": rows,
        "n_candidates": len(rows),
        "n_worth_introducing": len(introducible),
        "recommendation": (
            f"有 {len(introducible)} 个候选同时满足三项准入条件，可进入只读接入评估"
            if introducible else
            "无候选同时满足「A 股适配 + 离线可复现 + 许可证明确」："
            "按排期纪律**整阶段取消比强行引入更负责**"
        ),
        "affects_gate": False,
        "affects_signal": False,
        "note": ("评估不通过即取消阶段（排期已写明）；任何情况下 LLM 结论都不进信号路径。"),
    }


def offline_contrast(scores: Sequence[float], fwd_ret: Sequence[float],
                     notes: Optional[Sequence[Any]] = None,
                     *,
                     min_samples: int = 30) -> Dict[str, Any]:
    """离线对照（T20.3）：同一批样本上「有/无 LLM 附注」的命中率差异。

    关键结构性事实（必须如实写出来，而不是包装成"实验结果"）：
      投研附注**不参与**打分，因此两组用的是**同一份预测** ——
      命中率差异在结构上恒为 0。所以正确结论是
      ``unverifiable``（附注的效果在本项目链路里不可量化），并发起止损，
      而不是宣布"LLM 无提升"或"LLM 有提升"。

    若调用方提供了**带独立信号**的附注（``notes`` 中每条含 ``scores``），
    才会做真实对照；否则如实返回 unverifiable。
    """
    s = np.asarray(scores, dtype=float)
    r = np.asarray(fwd_ret, dtype=float)
    n = int(min(s.size, r.size))
    if n == 0:
        return {"kind": "research_assist_contrast", "available": False,
                "reason": "无样本", "affects_gate": False, "affects_signal": False}
    if n < min_samples:
        return {"kind": "research_assist_contrast", "available": False,
                "reason": f"样本不足（{n} < {min_samples}），不猜",
                "affects_gate": False, "affects_signal": False}

    s, r = s[:n], r[:n]
    from src.inference.ic import hit_rate

    base_hit = hit_rate(list(s), list(r))

    external_scores = None
    if notes:
        try:
            cand = [x.get("scores") for x in notes if isinstance(x, dict)
                    and x.get("scores") is not None]
            if len(cand) == n:
                external_scores = np.asarray(cand, dtype=float)
        except Exception:  # noqa: BLE001
            external_scores = None

    if external_scores is None:
        return {
            "kind": "research_assist_contrast",
            "generated_at": _now(),
            "available": True,
            "samples": n,
            "baseline_hit_rate": base_hit,
            "augmented_hit_rate": base_hit,
            "delta_hit_rate": 0.0,
            "verdict": "unverifiable",
            "conclusion": (
                "附注不参与打分，两组使用**同一份预测**，命中率差异在结构上恒为 0："
                "『LLM 附注是否提升命中率』在本项目链路里**不可量化**，"
                "按排期纪律记为不可验证并**止损**（不编造效果）"
            ),
            "stop_loss": True,
            "affects_gate": False,
            "affects_signal": False,
            "note": "附注只挂报告层，与信号/概率/门禁结构性不连通。",
        }

    aug_hit = hit_rate(list(external_scores), list(r))
    delta = None if (base_hit is None or aug_hit is None) else round(aug_hit - base_hit, 6)
    return {
        "kind": "research_assist_contrast",
        "generated_at": _now(),
        "available": True,
        "samples": n,
        "baseline_hit_rate": base_hit,
        "augmented_hit_rate": aug_hit,
        "delta_hit_rate": delta,
        "verdict": "measured",
        "conclusion": (
            f"使用了独立附注信号：命中率 {base_hit:.2%} → {aug_hit:.2%}"
            f"（差 {delta:+.4f}）；**仅作参考，未经多重比较校正不作达标证据**"
        ),
        "stop_loss": False,
        "affects_gate": False,
        "affects_signal": False,
        "note": "即便有独立信号，LLM 结论也不进信号路径（只读纪律）。",
    }


def attach_research_note(prediction: Dict[str, Any], note: str = "",
                         source: str = "") -> Dict[str, Any]:
    """把投研附注挂到**报告层**（T20.2）：类型上不可进入判据。

    附注字段独立命名（``research_note`` / ``research_source``），
    且本函数**不触碰**任何 ``probability`` / ``direction`` / ``signal`` 字段 ——
    这是只读纪律的结构性保证（而非口头约定）。
    """
    if not isinstance(prediction, dict):
        return prediction
    out = dict(prediction)
    out["research_note"] = str(note or "")
    out["research_source"] = str(source or "")
    out["research_note_readonly"] = True
    return out


def build_decision(research_evaluation: Dict[str, Any],
                   contrast: Optional[Dict[str, Any]] = None,
                   decided_by: str = "",
                   proceed: bool = False,
                   reason: str = "") -> Dict[str, Any]:
    """保留决策单（T20.4）：默认 defer / cancel，无人工签字不放行。"""
    worth = int(research_evaluation.get("n_worth_introducing", 0))
    unverifiable = bool((contrast or {}).get("verdict") == "unverifiable"
                        or (contrast or {}).get("stop_loss"))

    blockers: List[str] = []
    if worth == 0:
        blockers.append("无候选满足准入条件（A 股适配 + 离线可复现 + 许可证明确）")
    if unverifiable:
        blockers.append("离线对照不可量化（附注不参与打分），效果无法验证")
    if not str(decided_by or "").strip():
        blockers.append("缺少人工签字（decided_by）")

    if not str(decided_by or "").strip():
        verdict = VERDICT_CANCEL if worth == 0 else VERDICT_DEFER
        status = STATUS_PENDING
    else:
        verdict = VERDICT_PROCEED if (proceed and worth > 0) else VERDICT_CANCEL
        status = STATUS_CONFIRMED

    narrative = {
        VERDICT_CANCEL: ("本轮判定**取消**该链路（或不予引入）："
                         "按排期纪律，评估不通过则整阶段取消 —— 这比强行接入更负责"),
        VERDICT_DEFER: "缺乏人工签字或证据不足，维持 pending（不视为失败）",
        VERDICT_PROCEED: "人工签字确认保留为**只读**辅助链路（仍不进信号路径）",
    }[verdict]

    return {
        "kind": "research_assist_decision",
        "generated_at": _now(),
        "verdict": verdict,
        "status": status,
        "narrative": narrative,
        "blockers": blockers,
        "decided_by": str(decided_by or ""),
        "reason": str(reason or ""),
        "evidence": {
            "n_worth_introducing": worth,
            "contrast_verdict": (contrast or {}).get("verdict"),
            "recommendation": research_evaluation.get("recommendation"),
        },
        "affects_gate": False,
        "affects_signal": False,
        "note": ("无论结论如何，LLM 投研结论都只挂在报告层：不进信号路径、"
                 "不影响概率与门禁（``affects_signal``/``affects_gate`` 恒为 False）。"),
    }
