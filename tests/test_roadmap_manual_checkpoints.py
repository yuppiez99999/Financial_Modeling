"""人工检查点守卫：遗留检查点**不得自动通过**，决策包必须与排期一致（fail-close）。

设计要点（这是纪律测试，不是功能测试）：
  - 人工检查点**只能处于 `pending` 或 `confirmed`**：`confirmed` 必须带
    人工签署字段（confirmed_by / confirmed_at / decision），**任何 `completed`
    都是代签**——自动流程不得自行签字（2026-09-12 起用户可在 Issue 确认，见 §"确认即签字"）；
  - 人工检查点不得出现在 `auto_acceptable`：防选择自由度被自动吞掉；
  - `schedule/manual_checkpoints.json` 与 `plan.json` 必须**双向**一致：
    清单里的 id 必须真的是人工检查点、排期里的人工检查点必须在清单里；
  - 决策包文档必须存在且覆盖全部检查点 id；
  - 清单只允许 `status ∈ {pending, confirmed}` 且 `npc_may_decide=false`；
    条目一旦 `confirmed`，必须能追溯到**用户指令原文 + 签署人 + 时间 + 决策**——
    没有人工签字的 `confirmed` 与没有依据的 `completed` 一样是假进度。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = PROJECT_ROOT / "schedule" / "plan.json"
MANIFEST_PATH = PROJECT_ROOT / "schedule" / "manual_checkpoints.json"
DOC_PATH = PROJECT_ROOT / "00_kickoff" / "manual_checkpoints_round_g_h.md"

# 本轮（G 轮 + H 轮）必须收口的全部人工检查点。
# H 轮自动任务已全部交付（plan.json 各阶段 closing），故 S19/S20 的检查点
# （T19.4 / T20.1 / T20.4）**同样必须进决策包** —— 至此再无「已交付但未登记」。
REQUIRED_CHECKPOINT_IDS = {
    "T11.2", "T12.3", "T13.4", "T14.3", "T15.3", "T16.4", "T17.4", "T18.4",
    "T19.4", "T20.1", "T20.4",
}
# 已无「尚未开工」的阶段：H 轮全部开工（S16~S20 status=in_progress）。
# 保留空集仅为兼容旧签名（``include_not_started=False`` 与 True 等价）。
NOT_STARTED_STAGE_IDS: set = set()


def _plan() -> dict:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


# 本轮范围：plan.json 里**出现两次**的 S11~S14 属「旧轮 id 复用」，必须用
# 阶段名区分；本守卫只覆盖「高质量项目集成（G/H）」两轮 + S15~S18。
# 旧轮（多重收敛校正 / 特征扩充 / 试验登记 / 发布态检查）与 S1~S10 均不在范围。
_ROUND_MARKERS = ("G1", "G2", "G3", "G4", "G5", "H1", "H2", "H3")
_SCOPE_STAGE_IDS = {"S15", "S16", "S17", "S18", "S19", "S20"}


def _is_in_scope(stage: dict) -> bool:
    """本轮阶段 = S15~S20，或名称里带 G/H 轮标记的 S11~S14（排除旧轮同名阶段）。"""
    if stage["id"] in _SCOPE_STAGE_IDS:
        return True
    if stage["id"] in ("S11", "S12", "S13", "S14"):
        return any(m in (stage.get("name") or "") for m in _ROUND_MARKERS)
    return False


def _all_checkpoint_tasks(include_not_started: bool = True):
    """(stage, task) 对：本轮排期里被列为 manual_checkpoint 的任务。

    ``include_not_started=False`` 时跳过 S19/S20（尚未开工，其检查点属下一轮）。
    """
    out = []
    for stage in _plan()["stages"]:
        if not _is_in_scope(stage):
            continue
        if not include_not_started and stage["id"] in NOT_STARTED_STAGE_IDS:
            continue
        mc = set(stage.get("manual_checkpoint") or [])
        for task in stage["tasks"]:
            if task["id"] in mc:
                out.append((stage, task))
    return out


def _all_checkpoint_ids() -> set:
    return {t["id"] for _, t in _all_checkpoint_tasks()}


# ----------------------------------------------------------------------
# 不得自动通过
# ----------------------------------------------------------------------
# 人工检查点允许的状态：pending（待签）/ confirmed（已由人工签字确认）。
# completed **不在**其中——那意味着自动流程自行把「结论已出」当成「已采纳」。
CHECKPOINT_STATUSES = {"pending", "confirmed"}
DECISION_WORDS = {"keep", "defer", "reject", "cancel", "moot_by_convention"}


class TestNeverAutoPass:
    def test_no_manual_checkpoint_is_auto_completed(self):
        """人工检查点标 completed = 代签，永远不允许（fail-close）。"""
        bad = [t["id"] for _, t in _all_checkpoint_tasks()
               if t["status"] not in CHECKPOINT_STATUSES]
        assert not bad, f"人工检查点被自动完成（代签）: {bad}"

    def test_confirmed_checkpoints_carry_human_signature(self):
        """`confirmed` **必须**带人工签署字段：没有签字的 confirmed 就是假进度。"""
        for _, t in _all_checkpoint_tasks():
            if t["status"] != "confirmed":
                continue
            for key in ("confirmed_by", "confirmed_at", "confirmed_in", "decision"):
                assert t.get(key), f"{t['id']} 标 confirmed 却缺少 {key}（无签字依据）"
            assert t["decision"] in DECISION_WORDS, \
                f"{t['id']} 决策词表外取值: {t['decision']}"

    def test_confirmed_checkpoints_cannot_be_auto_run(self):
        """已确认的检查点仍不得被自动推进（确认 ≠ 交给自动流程执行）。"""
        for _, t in _all_checkpoint_tasks():
            if t["status"] == "confirmed":
                assert t.get("auto_run") is False, f"{t['id']} 被标 auto_run=true"

    def test_manual_checkpoint_not_in_auto_acceptable(self):
        for stage in _plan()["stages"]:
            overlap = set(stage.get("auto_acceptable") or []) & set(
                stage.get("manual_checkpoint") or [])
            assert not overlap, f"{stage['id']} 检查点混入自动项: {sorted(overlap)}"

    def test_manual_checkpoint_task_is_not_auto_run(self):
        for _, task in _all_checkpoint_tasks():
            assert task.get("auto_run") is False, \
                f"{task['id']} 是人工检查点却标 auto_run=true"

    def test_checkpoint_ids_have_no_completed_at_or_result(self):
        for _, task in _all_checkpoint_tasks():
            assert not task.get("completed_at"), f"{task['id']} 有 completed_at"
            assert not task.get("result"), f"{task['id']} 有 result（视为已决策）"


# ----------------------------------------------------------------------
# 清单一致性（双向）
# ----------------------------------------------------------------------
class TestManifestConsistency:
    def test_manifest_exists_and_policy_locks_decision(self):
        m = _manifest()
        assert m["policy"]["npc_may_decide"] is False
        assert set(m["policy"]["statuses_allowed"]) <= CHECKPOINT_STATUSES
        assert m["policy"].get("never_auto_filled") is True, \
            "清单必须声明「只能由人工指令置 confirmed」"

    def test_all_manifest_entries_in_allowed_status(self):
        for cp in _manifest()["checkpoints"]:
            assert cp["status"] in CHECKPOINT_STATUSES, \
                f"{cp['id']} 状态非法（completed 即代签）: {cp['status']}"

    def test_confirmed_manifest_entries_are_traceable(self):
        """已确认条目必须可追溯：签署人 + 时间 + 决策 + 依据 + 范围 + 证据。"""
        m = _manifest()
        for cp in m["checkpoints"]:
            if cp["status"] != "confirmed":
                continue
            for key in ("decision", "decision_basis", "script_must", "scope",
                        "out_of_scope", "confirmed_by", "confirmed_at"):
                assert cp.get(key), f"{cp['id']} 已确认却缺少 {key}"
            assert cp["decision"] in DECISION_WORDS, \
                f"{cp['id']} 决策词表外取值: {cp['decision']}"
            assert cp["evidence"], f"{cp['id']} 已确认却无证据引用"
        # 确认必须指向签发指令（Issue #40），否则属无据确认
        instr = m.get("instruction") or {}
        assert instr.get("issue"), "清单缺少 instruction.issue（确认来源）"
        assert instr.get("confirmed_by"), "清单缺少 instruction.confirmed_by"
        assert instr.get("text"), "清单缺少 instruction.text（指令原文）"

    def test_required_checkpoints_all_present(self):
        ids = {cp["id"] for cp in _manifest()["checkpoints"]}
        missing = REQUIRED_CHECKPOINT_IDS - ids
        assert not missing, f"决策包缺少遗留检查点: {sorted(missing)}"

    def test_manifest_ids_are_real_manual_checkpoints(self):
        # 「真实人工检查点」的判定是**全排期**口径：任一阶段 manual_checkpoint
        # 列出的任务 id 都是合法引用（I 轮 S21~S25 的规划期检查点同样有效）。
        real = {t["id"] for s in _plan()["stages"]
                for t in s["tasks"] if t["id"] in (s.get("manual_checkpoint") or [])}
        for cp in _manifest()["checkpoints"]:
            assert cp["id"] in real, f"{cp['id']} 不在 plan.json 的人工检查点里"

    def test_every_plan_manual_checkpoint_is_in_manifest(self):
        """反向：本轮排期里每个未收口的人工检查点都必须在决策包里。"""
        ids = {cp["id"] for cp in _manifest()["checkpoints"]}
        for _, task in _all_checkpoint_tasks(include_not_started=False):
            assert task["id"] in ids, f"{task['id']} 未进决策包"

    def test_manifest_entries_have_question_and_evidence_fields(self):
        for cp in _manifest()["checkpoints"]:
            for key in ("id", "stage", "priority", "question",
                        "current_default", "evidence", "doc", "status"):
                assert key in cp, f"{cp['id']} 缺少字段 {key}"
            assert cp["question"].strip(), f"{cp['id']} 问题为空"

    def test_priorities_are_unique(self):
        ps = [cp["priority"] for cp in _manifest()["checkpoints"]]
        assert len(ps) == len(set(ps)), "决策包优先级重复"

    def test_priorities_are_contiguous_from_one(self):
        """优先级必须 1..N 唯一连续：防「删一条留个洞」导致排序失真。"""
        ps = sorted(cp["priority"] for cp in _manifest()["checkpoints"])
        assert ps == list(range(1, len(ps) + 1)), f"优先级不连续: {ps}"

    def test_manifest_covers_all_h_round_checkpoints(self):
        """H 轮（S16~S20）检查点必须全部登记：防「已交付但未登记」。"""
        ids = {cp["id"] for cp in _manifest()["checkpoints"]}
        h_ids = {t["id"] for st, t in _all_checkpoint_tasks()
                 if st["id"] in ("S16", "S17", "S18", "S19", "S20")}
        missing = h_ids - ids
        assert not missing, f"H 轮人工检查点未进决策包: {sorted(missing)}"


class TestNoDanglingEvidenceRefs:
    """决策材料不得悬空：清单引用的文档/证据**必须真实存在**。

    S19/S20 那轮的教训 —— H4/H5 的结论文档被清单引用但文件根本不存在，
    等于「决策材料缺失却对外宣称材料齐备」。
    """

    def _resolve(self, ref: str) -> Path:
        """把 `路径#锚点` 解析为工程内绝对路径（锚点只校验文件，不校验小节）。"""
        return PROJECT_ROOT / ref.split("#", 1)[0]

    def test_doc_refs_exist(self):
        bad = [cp["id"] for cp in _manifest()["checkpoints"]
               if not self._resolve(cp["doc"]).is_file()]
        assert not bad, f"决策包 doc 引用悬空（文件不存在）: {bad}"

    def test_evidence_refs_exist_or_declared_optional(self):
        """证据文件必须存在，除非显式登记进 ``KNOWN_OPTIONAL_EVIDENCE``。

        可选证据 =「需按需生成、不入 git」的产物（如按需跑命令才出的报告）。
        不允许默默缺失 —— 缺失必须在清单里被点名列明。
        """
        optional = set(_manifest().get("known_optional_evidence") or [])
        bad = []
        for cp in _manifest()["checkpoints"]:
            for ev in cp["evidence"]:
                resolved = self._resolve(ev)
                if resolved.exists():
                    continue
                if any(ev == o or ev.split("#", 1)[0] == o.split("#", 1)[0]
                       for o in optional):
                    continue
                bad.append(f"{cp['id']} -> {ev}")
        assert not bad, f"证据引用悬空且未声明为可选项: {bad}"

    def test_known_optional_evidence_entries_are_nonempty(self):
        for item in _manifest().get("known_optional_evidence") or []:
            assert str(item).strip(), "known_optional_evidence 含空条目"


class TestHRoundStageClosing:
    """H 轮（S16~S20）阶段收口字段必须齐备且风格一致（防漏补）。"""

    _H_STAGES = ("S16", "S17", "S18", "S19", "S20")

    def _h_stages(self):
        return [s for s in _plan()["stages"] if s["id"] in self._H_STAGES]

    def test_all_h_stages_have_closing_fields(self):
        for s in self._h_stages():
            for key in ("round", "auto_acceptable_completed_at",
                        "auto_acceptable_verified", "closing"):
                assert key in s, f"{s['id']} 缺收口字段 {key}"

    def test_h_round_labels_are_h1_to_h5(self):
        got = {s["id"]: s["round"] for s in self._h_stages()}
        assert got == {"S16": "H1", "S17": "H2", "S18": "H3",
                       "S19": "H4", "S20": "H5"}, f"H 轮编号错位: {got}"

    def test_closing_scope_declares_gate_false(self):
        for s in self._h_stages():
            assert s["closing"]["affects_gate"] is False, \
                f"{s['id']} closing 未声明 affects_gate=false"
            assert s["closing"].get("auto_scope"), f"{s['id']} closing 缺 auto_scope"
            assert s["closing"].get("evidence"), f"{s['id']} closing 缺 evidence"

    def test_auto_completed_stage_status_is_backed_by_signature(self):
        """自动部分交付 ≠ 阶段完成：无人工签字不得标 completed。

        两种合法状态，二选一（与 G 轮同口径）：
        - ``in_progress``：自动部分已交付、人工检查点仍 ``pending``（未签字）；
        - ``completed``：自动部分已交付 **且** 人工检查点已 ``confirmed``
          （``closing.manual_scope`` 声明已确认 + 阶段带 ``confirmed_by`` / ``confirmed_at``）。

        自动交付本身**不构成**阶段 ``completed`` 的依据 —— 这正是防「假进度」的钉子。
        """
        for s in self._h_stages():
            assert s["status"] in ("in_progress", "completed"), \
                f"{s['id']} status 非法: {s['status']}"
            assert s["closing"]["manual_scope"], f"{s['id']} closing 缺 manual_scope"
            if s["status"] == "completed":
                assert "已确认" in s["closing"]["manual_scope"], \
                    f"{s['id']} 标 completed 但 manual_scope 未声明已确认"
                assert s.get("confirmed_by") and s.get("confirmed_at"), \
                    f"{s['id']} 标 completed 却缺签署字段（疑似代签）"


# ----------------------------------------------------------------------
# 决策包文档
# ----------------------------------------------------------------------
class TestDecisionDoc:
    def test_doc_exists(self):
        assert DOC_PATH.exists(), "缺少遗留检查点决策包文档"

    def test_doc_covers_all_checkpoints(self):
        doc = DOC_PATH.read_text(encoding="utf-8")
        for cp_id in REQUIRED_CHECKPOINT_IDS:
            assert cp_id in doc, f"决策包文档未覆盖 {cp_id}"

    def test_doc_covers_h_round_checkpoints(self):
        """H 轮（含 S19/S20）检查点必须在决策包文档里有独立小节。"""
        doc = DOC_PATH.read_text(encoding="utf-8")
        for cp_id in ("T19.4", "T20.1", "T20.4"):
            assert cp_id in doc, f"决策包文档未覆盖 H 轮检查点 {cp_id}"

    def test_doc_states_no_npc_decision(self):
        doc = DOC_PATH.read_text(encoding="utf-8")
        assert "不代签" in doc, "决策包必须显式声明不代签"
        assert "affects_gate" in doc, "决策包必须声明 affects_gate 纪律"

    def test_checkpoints_stay_readonly(self):
        """本轮零门禁改动：相关阶段 note/任务描述不得出现放宽门禁表述。"""
        forbidden = ["降低门禁", "放宽门禁", "下调门禁", "改低门禁"]
        for stage in _plan()["stages"]:
            if not _is_in_scope(stage):
                continue
            blob = stage["name"] + (stage.get("note") or "") + " ".join(
                t["name"] for t in stage["tasks"])
            for word in forbidden:
                assert word not in blob, f"{stage['id']} 出现门禁放宽表述: {word}"
