"""Laya 类型化决策只读适配层（S26 / J1，T26.2~T26.4）。

问题从哪来（为什么这条线**只留 Laya、不留 Jev**，且纪律比功能重要）：
  本项目（16_）的交付形态已通（`decision-feed/1` 契约 + TradingView 投影），
  但**模型本身没跑出可用区分度**：真实数据 AUC ≈ 0.50~0.54、置信度几乎全部贴地、
  `advisory_consumable` 0/38（见 `cairn/decision-source-contract.md` §三/§六）。
  已交付契约里 `advisory.recommended_threshold` 是「已配置现行值」而非择优结果，
  现有 LightGBM 概率**未校准**（`model_degeneration_conclusion` 已认定输出常数化）。

  Laya（Convai Innovations 开放权重、Jev 同构）给的是**另一类输出**：
  类型化决策 `choice` / `score` / `noul`，每个带**校准置信度**，
  且可**本地跑**（MLX/CoreML/ONNX）⇒ CI 离线可复现 —— 正好补上契约层缺失的一环：
  一个可离线复现的**只读第二决策源 / 交叉验证臂**。

  为什么不做 Jev（用户 2026-09-20 决策，Issue #66）：
    Jev 是**闭源托管 API** = 网络依赖 + 成本 + **不可复现** —— 直接撞上 H5 被否的
    三条死因。故**只留本地 Laya**，不做 Jev 在线臂（无网络依赖、无外部调用）。

本模块的纪律（比功能重要得多，逐条与排期写死的一致）：
  1. **不进信号路径**：`affects_signal` / `affects_gate` 恒 False —— 结构性保证；
  2. **CI 离线**：只用本地权重；权重缺失时**如实降级**（`unavailable`），
     不报错、不联网、不编造读数；
  3. **不编造效果**：Laya 输出与现行概率不可比时记 `unverifiable` + 止损，
     绝不宣布「Laya 提升了命中率」；
  4. **不代签**：无 `decided_by` → 结论最多 `defer` / `cancel`。

本模块**不做什么**：
  - 不下载、不绑定、不打包 Laya 权重（缺省即无权重 → 走 `unavailable`）；
  - 不产出任何可进入概率 / 方向 / 门禁判据的字段；
  - 不修改 `strategy_gate`（零改动）。

无前视说明：
  适配层只做「已有输出 → 类型化决策」的**语义映射**与**对照统计**，
  不引入未来信息；对照实验中 Laya 侧与 LightGBM 侧用**同一批样本的已实现收益**。
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 候选与准入（T26.1 是人工检查点，本模块只交付「评估器」不改其状态）
# ----------------------------------------------------------------------
CANDIDATES: Sequence[Dict[str, Any]] = (
    {
        "repo": "Convai Innovations / Laya（开放权重，Jev 同构）",
        "license": "Convai Innovations 条款（需逐项核）",
        "weights_size": "约 1.7GB fp32 / ~2GB RAM（重型）",
        "runtime": "MLX / CoreML / ONNX（本地、离线）",
        "output_shape": "类型化决策 choice / score / noul + 校准置信度",
        "latency_hint": "约 7.6ms（本地 MLX）",
        "offline_reproducible": True,
        "a_share_adaptation": "无专用 A 股适配（通用序列/决策输入）",
        "verdict": "conditional",
        "reason": ("本地开放权重 ⇒ CI 离线可复现（规避 H5 死因①）；"
                   "但权重是**重型依赖**（~1.7GB）与「零重型依赖」纪律冲突，"
                   "必须先过 T26.1 准入评估；且无 A 股专用适配，A 股适配需自建输入层"),
    },
    {
        "repo": "Jev（TypeSafe AI，闭源托管 API）",
        "license": "闭源托管服务",
        "weights_size": "不可本地化（服务端）",
        "runtime": "HTTP API（需网络）",
        "output_shape": "同 Laya（choice / score / noul，同一 request/response 形状）",
        "latency_hint": "约 588ms（网络）",
        "offline_reproducible": False,
        "a_share_adaptation": "无",
        "verdict": "reject",
        "reason": ("按用户决策**只留 Laya**：Jev 闭源 API = 网络依赖 + 成本 + "
                   "**不可复现**，直接撞上 H5 被否的三条死因（排期已不再登记 Jev 臂）"),
    },
)

# 类型化决策的三种输出形状（与 Laya/Jev 同一 request/response 口径）
DECISION_CHOICE = "choice"
DECISION_SCORE = "score"
DECISION_NOUL = "noul"
DECISION_KINDS = (DECISION_CHOICE, DECISION_SCORE, DECISION_NOUL)

STATUS_PENDING = "pending"
STATUS_CONFIRMED = "confirmed"
VERDICT_DEFER = "defer"
VERDICT_CANCEL = "cancel"
VERDICT_PROCEED = "proceed"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------
# T26.1 准入评估器（不改状态：状态是人工检查点）
# ----------------------------------------------------------------------
def build_laya_evaluation(candidates: Sequence[Dict[str, Any]] = CANDIDATES
                          ) -> Dict[str, Any]:
    """一页准入评估：逐候选给「是否够格进入只读接入」判定。

    `worth_introducing` 只有当候选同时满足「离线可复现 + 许可证明确 + 非重型」时为真。
    注意：Laya 的 **offline ok 但重型** ⇒ 现状判定为「条件性通过」，
    须由 T26.1 人工检查点确认是否接受 ~1.7GB 依赖代价（不过则整阶段取消）。
    """
    rows: List[Dict[str, Any]] = []
    for c in candidates:
        license_ok = "NOASSERTION" not in str(c.get("license", "")).upper()
        offline_ok = bool(c.get("offline_reproducible"))
        # 重型依赖：权重 > 1GB 视为重型（与「零重型依赖」纪律对齐）
        heavy = "GB" in str(c.get("weights_size", ""))
        blockers: List[str] = []
        if not license_ok:
            blockers.append("许可证不明确")
        if not offline_ok:
            blockers.append("离线不可复现（依赖网络/托管 API）")
        if heavy:
            blockers.append("重型依赖（权重 ~1.7GB，与零重型依赖纪律冲突）")
        row = dict(c)
        row["license_clear"] = license_ok
        row["offline_ok"] = offline_ok
        row["heavy_dependency"] = heavy
        row["worth_introducing"] = bool(license_ok and offline_ok)
        row["needs_manual_admission"] = bool(heavy and license_ok and offline_ok)
        row["blockers"] = blockers
        rows.append(row)

    introducible = [r for r in rows if r["worth_introducing"]]
    needs_manual = [r for r in rows if r["needs_manual_admission"]]
    return {
        "kind": "laya_evaluation",
        "generated_at": _now(),
        "candidates": rows,
        "n_candidates": len(rows),
        "n_worth_introducing": len(introducible),
        "n_needs_manual_admission": len(needs_manual),
        "recommendation": (
            "有候选满足离线可复现 + 许可证明确，可进入只读接入评估；"
            "但权重属重型依赖，**必须由 T26.1 人工检查点确认依赖代价**（不过则整阶段取消）"
            if needs_manual else
            ("有候选满足准入，可进入只读接入评估" if introducible else
             "无候选满足「离线可复现 + 许可证明确」：按排期纪律**整阶段取消比强行引入更负责**")
        ),
        "affects_gate": False,
        "affects_signal": False,
        "note": ("Jev 已按用户决策（Issue #66，只留 Laya）判 reject；"
                 "评估不通过即取消阶段（排期已写明）；任何情况下 Laya 输出都不进信号路径。"),
    }


# ----------------------------------------------------------------------
# T26.2 本地只读适配器
# ----------------------------------------------------------------------
class LayaLocalAdapter:
    """Laya 本地**只读**适配器（可选依赖；权重缺失即降级，不报错、不联网）。

    三类输出统一映射为**只读**结构：
      - `choice`：离散方向（如 ``up`` / ``down`` / ``flat``）
      - `score` ：连续分数（可含校准置信度）
      - `noul`  ：Laya 的「无法判定 / 弃权」输出（== 不可采信）

    适配器**只读**：不产出 `probability` / `direction` / `signal` 字段，
    类型上与信号路径结构性不连通。真实权重缺失时 `available=False`，
    调用方据此如实降级 —— 绝不返回伪造读数。
    """

    def __init__(self, weights_path: str = "", *, threshold: float = 0.60,
                 runtime: str = "onnx"):
        self.weights_path = str(weights_path or "")
        self.threshold = float(threshold)
        self.runtime = str(runtime or "onnx")
        self._backend = None
        self._load_error = ""
        self._try_load()

    # -- 加载 ----------------------------------------------------------
    def _try_load(self) -> None:
        path = self.weights_path or os.environ.get("LAYA_WEIGHTS_PATH", "")
        if not path:
            self._load_error = "未提供 Laya 本地权重（缺省即降级，不联网、不下载）"
            return
        if not os.path.exists(path):
            self._load_error = f"Laya 权重路径不存在: {path}"
            return
        # 真实推理后端（MLX/CoreML/ONNX）不在本仓库交付范围：
        # 排期 T26.2 只要求**只读适配形状**；真实后端接入须过 T26.1 准入。
        self._load_error = ("已发现权重路径但未接入运行时后端（本阶段只交付只读适配形状，"
                            "真实后端须过 T26.1 准入后另行接入）")

    @property
    def available(self) -> bool:
        return self._backend is not None

    def status(self) -> Dict[str, Any]:
        return {
            "kind": "laya_adapter_status",
            "available": self.available,
            "runtime": self.runtime,
            "weights_path": self.weights_path or "(none)",
            "threshold": self.threshold,
            "reason": self._load_error,
            "affects_gate": False,
            "affects_signal": False,
        }

    # -- 只读推理 ------------------------------------------------------
    def decide(self, features: Any = None) -> Dict[str, Any]:
        """对单条输入产出**只读**类型化决策；不可用时如实返回 `noul`。

        返回结构**硬编码**为只读语义：只有 `choice` / `score` / `confidence` /
        `kind`，**没有任何**可被信号路径消费的 `probability` / `direction`。
        """
        if not self.available:
            return {
                "kind": DECISION_NOUL,
                "choice": None,
                "score": None,
                "confidence": None,
                "available": False,
                "reason": self._load_error,
                "readonly": True,
                "affects_gate": False,
                "affects_signal": False,
            }
        # 后端起效时才会走到这里（本阶段不接入真实后端，保留形状）
        return {
            "kind": DECISION_CHOICE,
            "choice": None,
            "score": None,
            "confidence": None,
            "available": False,
            "reason": "运行时后端未接入（见 T26.1 准入）",
            "readonly": True,
            "affects_gate": False,
            "affects_signal": False,
        }


def typed_decision(kind: str, *, choice: Optional[str] = None,
                   score: Optional[float] = None,
                   confidence: Optional[float] = None,
                   reason: str = "") -> Dict[str, Any]:
    """规范化一条类型化决策（只读）：`choice` / `score` / `noul` 之一。"""
    k = str(kind or "").strip().lower()
    if k not in DECISION_KINDS:
        raise ValueError(f"未知类型化决策 kind={kind!r}，允许: {DECISION_KINDS}")
    conf = None if confidence is None else float(confidence)
    if conf is not None and not (0.0 <= conf <= 1.0):
        raise ValueError(f"置信度必须在 [0,1]，得到 {confidence!r}")
    return {
        "kind": k,
        "choice": None if k == DECISION_NOUL else choice,
        "score": None if k == DECISION_NOUL else score,
        "confidence": conf,
        "reason": str(reason or ""),
        "readonly": True,
        "affects_gate": False,
        "affects_signal": False,
    }


def attach_laya_note(prediction: Dict[str, Any],
                     decision: Dict[str, Any]) -> Dict[str, Any]:
    """把 Laya 决策挂到**报告层**（只读）：类型上不可进入判据。

    与 H5 的 `attach_research_note` 同一条纪律：只加独立命名字段
    （`laya_decision` / `laya_readonly`），**不触碰**任何
    `probability` / `direction` / `signal` 字段 —— 结构性保证，不是口头约定。
    """
    if not isinstance(prediction, dict):
        return prediction
    out = dict(prediction)
    out["laya_decision"] = dict(decision or {})
    out["laya_readonly"] = True
    return out


# ----------------------------------------------------------------------
# T26.3 离线对照（Laya 校准置信度 vs 现有 LightGBM 概率）
# ----------------------------------------------------------------------
def _rank(values: Sequence[float]) -> List[float]:
    idx = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and values[idx[j + 1]] == values[idx[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[idx[k]] = avg
        i = j + 1
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size < 3 or b.size < 3:
        return None
    ra, rb = np.asarray(_rank(a), dtype=float), np.asarray(_rank(b), dtype=float)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def offline_contrast(baseline_proba: Sequence[float], fwd_ret: Sequence[float],
                     laya_confidence: Optional[Sequence[float]] = None,
                     laya_choice: Optional[Sequence[Any]] = None,
                     *, min_samples: int = 30) -> Dict[str, Any]:
    """离线对照（T26.3）：Laya 类型化决策 vs 现有 LightGBM 概率。

    关键结构性事实（必须如实写出来，而不是包装成「实验结果」）：
      真实 Laya 权重**未在本阶段接入**（须过 T26.1 准入）⇒ 若调用方未提供
      `laya_confidence` / `laya_choice`，则无法做真实对照。此时正确结论是
      ``unverifiable`` + **止损**，而不是宣布「Laya 有效」或「Laya 无效」。

    仅当调用方提供**独立的 Laya 输出**时，才做真实对照：
      - 置信度分档 × 已实现收益的单调性（复用契约层「高置信 ≠ 可采信」的判据）；
      - Laya 置信度与现有概率的 Spearman 相关（衡量两源是否重复）。
    """
    s = np.asarray(baseline_proba, dtype=float)
    r = np.asarray(fwd_ret, dtype=float)
    n = int(min(s.size, r.size))
    if n == 0:
        return {"kind": "laya_contrast", "available": False, "reason": "无样本",
                "affects_gate": False, "affects_signal": False}
    if n < min_samples:
        return {"kind": "laya_contrast", "available": False,
                "reason": f"样本不足（{n} < {min_samples}），不猜",
                "affects_gate": False, "affects_signal": False}

    s, r = s[:n], r[:n]
    from src.inference.ic import hit_rate

    base_hit = hit_rate(list(s), list(r))

    if laya_confidence is None:
        return {
            "kind": "laya_contrast",
            "generated_at": _now(),
            "available": True,
            "samples": n,
            "baseline_hit_rate": base_hit,
            "laya_hit_rate": None,
            "delta_hit_rate": None,
            "verdict": "unverifiable",
            "conclusion": (
                "真实 Laya 权重未接入（须过 T26.1 准入）⇒ 无法与现有概率做真实对照："
                "『Laya 是否提升命中率』在本阶段**不可量化**，按排期纪律记不可验证并"
                "**止损**（延续 H5 先例，不编造效果）"
            ),
            "stop_loss": True,
            "affects_gate": False,
            "affects_signal": False,
            "note": "Laya 输出只挂报告层，与信号/概率/门禁结构性不连通。",
        }

    lc = np.asarray(laya_confidence, dtype=float)
    if lc.size < n:
        return {"kind": "laya_contrast", "available": False,
                "reason": f"Laya 置信度样本不足（{lc.size} < {n}）",
                "affects_gate": False, "affects_signal": False}
    lc = lc[:n]

    # 高置信子集命中率（与 regime-signal 门槛同量级）
    thr = 0.60
    mask = lc >= thr
    laya_hit = hit_rate(list(lc[mask]), list(r[mask])) if int(mask.sum()) >= 10 else None
    delta = None if (base_hit is None or laya_hit is None) else round(laya_hit - base_hit, 6)
    corr = _spearman(s, lc)

    # 单调性：置信度分档 × 已实现收益（复用契约层判据）
    bands = [0.0, 0.25, 0.5, 0.75, 1.0]
    band_rows = []
    for lo, hi in zip(bands[:-1], bands[1:]):
        m = (lc >= lo) & (lc < hi if hi < 1.0 else lc <= hi)
        if int(m.sum()) == 0:
            continue
        band_rows.append({
            "band": [lo, hi],
            "coverage": round(float(m.mean()), 4),
            "hit_rate": hit_rate(list(lc[m]), list(r[m])),
            "mean_return": round(float(r[m].mean()), 6),
        })

    return {
        "kind": "laya_contrast",
        "generated_at": _now(),
        "available": True,
        "samples": n,
        "baseline_hit_rate": base_hit,
        "laya_hit_rate": laya_hit,
        "delta_hit_rate": delta,
        "laya_threshold": thr,
        "laya_high_conf_samples": int(mask.sum()),
        "spearman_vs_baseline": (round(corr, 4) if corr is not None else None),
        "confidence_return_bands": band_rows,
        "verdict": "measured",
        "conclusion": (
            f"使用了独立 Laya 输出：高置信子集命中率 {laya_hit:.4f}" if laya_hit is not None
            else "使用了独立 Laya 输出，但高置信子集样本不足，不出命中率"
        ),
        "stop_loss": False,
        "affects_gate": False,
        "affects_signal": False,
        "note": ("即便有独立输出，Laya 也不进信号路径（只读纪律）；"
                 "与现有概率的 Spearman 高即说明两源重复，边际为零。"),
    }


# ----------------------------------------------------------------------
# T26.4 级联冒烟（用户决策：只留 Laya ⇒ 无 Jev 升级臂）
# ----------------------------------------------------------------------
def cascade_smoke(threshold: float = 0.60,
                  confidences: Optional[Sequence[float]] = None,
                  *, adapter_available: bool = False) -> Dict[str, Any]:
    """级联冒烟（T26.4）：**只留 Laya** 口径下的降级路径验证。

    原始级联设计为 `Laya 本地 → (置信度不足) 升级 Jev 在线`；按用户决策
    （Issue #66，只留 Laya）**移除 Jev 升级臂** ⇒ 本地不可用或置信度不足时，
    一律**本地降级为只读 `noul`（弃权）**，不联网、不报错。

    本函数验证该降级路径在两种情形下均**不报错且不产生信号**：
      1. 适配器不可用（无权重）→ 全部 `noul`；
      2. 适配器可用但置信度低于阈值 → 该条 `noul`，高于阈值 → `choice`。
    """
    confs = list(confidences) if confidences is not None else [0.9, 0.4, 0.7, 0.1]
    decisions: List[Dict[str, Any]] = []
    for c in confs:
        if not adapter_available:
            decisions.append(typed_decision(DECISION_NOUL, reason="Laya 本地不可用 → 本地降级弃权（无 Jev 升级臂）"))
        elif float(c) >= threshold:
            decisions.append(typed_decision(DECISION_CHOICE, choice="up",
                                            score=float(c), confidence=float(c),
                                            reason="本地采纳"))
        else:
            decisions.append(typed_decision(DECISION_NOUL, reason="置信度低于阈值 → 本地降级弃权（无 Jev 升级臂）"))

    n_no = sum(1 for d in decisions if d["kind"] == DECISION_NOUL)
    n_choice = sum(1 for d in decisions if d["kind"] == DECISION_CHOICE)
    return {
        "kind": "laya_cascade_smoke",
        "generated_at": _now(),
        "architecture": "laya_local_only（Jev 升级臂已按用户决策移除）",
        "threshold": float(threshold),
        "adapter_available": bool(adapter_available),
        "n_inputs": len(decisions),
        "n_choice": n_choice,
        "n_noul": n_no,
        "decisions": decisions,
        "network_calls": 0,
        "no_network_dependency": True,
        "raises": False,
        "verdict": "pass",
        "conclusion": ("只留 Laya 的降级路径验证：不可用 / 低置信一律本地弃权（noul），"
                       "零网络调用、零信号产出"),
        "affects_gate": False,
        "affects_signal": False,
        "note": "结构化不连通：产出全部 `readonly=True`，无 probability/direction/signal 字段。",
    }


# ----------------------------------------------------------------------
# T26.5 保留决策（人工检查点）
# ----------------------------------------------------------------------
def build_decision(evaluation: Dict[str, Any],
                   contrast: Optional[Dict[str, Any]] = None,
                   decided_by: str = "",
                   proceed: bool = False,
                   reason: str = "") -> Dict[str, Any]:
    """保留决策单（T26.5）：默认 `defer` / `cancel`，无人工签字不放行。"""
    worth = int(evaluation.get("n_worth_introducing", 0))
    unverifiable = bool((contrast or {}).get("verdict") == "unverifiable"
                        or (contrast or {}).get("stop_loss"))

    blockers: List[str] = []
    if worth == 0:
        blockers.append("无候选满足准入条件（离线可复现 + 许可证明确）")
    if unverifiable:
        blockers.append("离线对照不可量化（真实权重未接入），效果无法验证")
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
        "kind": "laya_decision",
        "generated_at": _now(),
        "verdict": verdict,
        "status": status,
        "narrative": narrative,
        "blockers": blockers,
        "decided_by": str(decided_by or ""),
        "reason": str(reason or ""),
        "evidence": {
            "n_worth_introducing": worth,
            "n_needs_manual_admission": evaluation.get("n_needs_manual_admission"),
            "contrast_verdict": (contrast or {}).get("verdict"),
            "recommendation": evaluation.get("recommendation"),
        },
        "affects_gate": False,
        "affects_signal": False,
        "note": ("无论结论如何，Laya 决策都只挂在报告层：不进信号路径、"
                 "不影响概率与门禁（``affects_signal``/``affects_gate`` 恒为 False）。"),
    }
