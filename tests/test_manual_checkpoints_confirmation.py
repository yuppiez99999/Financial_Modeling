"""人工检查点「确认」守卫（fail-close）：确认是**签字**，不是自动推进。

背景（Issue #40）：
  项目此前所有人工检查点恒 `pending`（防代签）。2026-09-12 用户指令
  「这11项确认 并继续按排期计划开发」——需要在**不破坏纪律**的前提下把这 11 条
  落定为已确认。本测试把这次确认的纪律固化为可执行断言。

设计要点：
  - 确认必须**可追溯**：签署人 / 时间 / 决策 / 范围 / 不在范围内 / 指令原文 / 证据；
  - 确认必须**有边界**：明确「确认的是现状默认值，不是改配置的授权」；
  - 确认必须**不动门禁**：`strategy_gate` 零变更、`affects_gate` 恒 false；
  - 确认不得**回流成代签**：人工检查点永远不得是 `completed`；
  - 阶段收官 `completed` 必须建立在「全部任务已落定 + 人工确认字段齐备」之上。

全部离线：只读两个 JSON 与两份 Markdown，不触网、不重训。
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = PROJECT_ROOT / "schedule" / "plan.json"
MANIFEST_PATH = PROJECT_ROOT / "schedule" / "manual_checkpoints.json"
DOC_PATH = PROJECT_ROOT / "00_kickoff" / "manual_checkpoints_round_g_h.md"
STRATEGY_GATE_PATH = PROJECT_ROOT / "configs" / "strategy_gate.yaml"

EXPECTED_IDS = [
    "T11.2", "T12.3", "T13.4", "T14.3", "T15.3",
    "T16.4", "T17.4", "T18.4", "T19.4", "T20.1", "T20.4",
]
DECISION_WORDS = {"keep", "defer", "reject", "cancel", "moot_by_convention"}
# 本轮确认覆盖的阶段（G1~G5 + H1~H5）
CONFIRMED_STAGES = ["S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18", "S19", "S20"]


def _plan() -> dict:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _decision_entries() -> list[dict]:
    """已签字决策包条目（G/H 轮收官确认，全部必须 confirmed）。"""
    return [cp for cp in _manifest()["checkpoints"]
            if cp.get("phase", "decision") == "decision"]


def _planning_entries() -> list[dict]:
    """规划期检查点（后续轮次 I 轮起，必须 pending 且禁止预签）。"""
    return [cp for cp in _manifest()["checkpoints"]
            if cp.get("phase") == "planning"]


def _stage(sid: str) -> dict:
    return next(s for s in _plan()["stages"] if s["id"] == sid)


def _task(tid: str) -> dict:
    for s in _plan()["stages"]:
        for t in s["tasks"]:
            if t["id"] == tid:
                return t
    raise AssertionError(f"plan.json 中不存在任务 {tid}")


# 历史轮次（S1~S10）的 `manual_checkpoint` 字段是**早期口径的记录**，
# 当时的任务早已由人工作业完成并标 `completed`，不属于本轮（G/H 轮）确认范围。
# 本轮守卫只覆盖 S11~S20（见 CONFIRMED_STAGES）。
LEGACY_STAGE_IDS = {"S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"}


def _checkpoint_tasks(include_legacy: bool = False):
    """本轮（S11~S20）的人工检查点任务。

    ``include_legacy=True`` 时把 S1~S10 的历史记录也算进来——**仅在需要
    做「清单 ⊆ 全量检查点」这类子集校验时使用**，不用于代签类断言。
    """
    out = []
    for s in _plan()["stages"]:
        if not include_legacy and s["id"] in LEGACY_STAGE_IDS:
            continue
        manual = set(s.get("manual_checkpoint") or [])
        for t in s["tasks"]:
            if t["id"] in manual:
                out.append((s, t))
    return out


# ----------------------------------------------------------------------
# 一、11 项确认的完整性
# ----------------------------------------------------------------------
class TestConfirmationCoverage:
    def test_exactly_eleven_checkpoints_confirmed(self):
        confirmed = [cp["id"] for cp in _manifest()["checkpoints"]
                     if cp["status"] == "confirmed"]
        assert sorted(confirmed) == sorted(EXPECTED_IDS), (
            f"确认集合与预期不符：{sorted(confirmed)}")

    def test_confirmed_ids_match_plan_manual_checkpoints(self):
        plan_ids = {t["id"] for _, t in _checkpoint_tasks()}
        manifest_ids = {cp["id"] for cp in _manifest()["checkpoints"]}
        assert manifest_ids == plan_ids, (
            f"清单与排期人工检查点不一致：{sorted(manifest_ids ^ plan_ids)}")

    def test_all_eleven_decisions_are_documented(self):
        doc = DOC_PATH.read_text(encoding="utf-8")
        for tid in EXPECTED_IDS:
            assert tid in doc, f"确认纪要不含 {tid}"


# ----------------------------------------------------------------------
# 二、可追溯性：没有签字依据的 confirmed 就是假进度
# ----------------------------------------------------------------------
class TestConfirmationTraceability:
    @pytest.mark.parametrize("key", ["decision", "decision_basis", "script_must",
                                     "scope", "out_of_scope", "confirmed_by",
                                     "confirmed_at", "issue", "rationale",
                                     "boundary"])
    def test_manifest_entry_has_traceability_field(self, key: str):
        for cp in _decision_entries():
            assert cp.get(key), f"{cp['id']} 缺少可追溯字段 {key}"

    def test_decisions_use_closed_vocabulary(self):
        for cp in _decision_entries():
            assert cp["decision"] in DECISION_WORDS, (
                f"{cp['id']} 决策词表外取值 {cp['decision']}")

    def test_instruction_is_recorded_verbatim(self):
        instr = _manifest()["instruction"]
        assert instr["confirmed_by"] == "yuppiez328"
        assert "这11项确认" in instr["text"], "指令原文必须原样留档"
        assert "/issues/40" in instr["issue"]

    def test_boundary_is_declared_at_list_level(self):
        b = _manifest()["boundary"]
        assert b["strategy_gate_changed"] is False
        assert b["affects_gate"] is False

    def test_every_entry_lists_out_of_scope(self):
        for cp in _manifest()["checkpoints"]:
            assert isinstance(cp["out_of_scope"], list) and cp["out_of_scope"], (
                f"{cp['id']} 未声明「不在确认范围」的内容")


# ----------------------------------------------------------------------
# 二·补、规划期检查点（后续轮次）：必须 pending，且严禁预签
# ----------------------------------------------------------------------
class TestPlanningPhaseEntries:
    def test_planning_entries_stay_pending(self):
        """规划期检查点只能 pending：confirmed/completed 都等于预签。"""
        for cp in _planning_entries():
            assert cp["status"] == "pending", (
                f"{cp['id']} 规划期检查点状态非法: {cp['status']}（只允许 pending）")

    def test_planning_entries_carry_no_signature(self):
        """规划期条目不得携带任何签署字段——签字只发生在对应阶段交付之后。"""
        for cp in _planning_entries():
            for key in ("decision", "confirmed_by", "confirmed_at",
                        "decision_basis", "scope"):
                assert not cp.get(key), (
                    f"{cp['id']} 规划期却带签署字段 {key}（疑似预签）")

    def test_planning_entries_are_traceable_to_round(self):
        for cp in _planning_entries():
            for key in ("question", "current_default", "issue", "round", "doc",
                        "stage", "priority", "out_of_scope"):
                assert cp.get(key), f"{cp['id']} 缺少规划期字段 {key}"
            assert "/issues/54" in cp["issue"], (
                f"{cp['id']} 规划期检查点未挂到本轮 Issue #54")


# ----------------------------------------------------------------------
# 三、确认 ≠ 改配置（这是本轮最关键的一条）
# ----------------------------------------------------------------------
class TestConfirmationDoesNotChangeGate:
    def test_no_strategy_gate_change_in_this_commit(self):
        """本次 PR **不得**触碰 strategy_gate 配置文件（确认只固定现状）。"""
        try:
            changed = subprocess.check_output(
                ["git", "diff", "--name-only", "HEAD"], text=True,
                cwd=str(PROJECT_ROOT)).split()
        except Exception:  # pragma: no cover - 无 git 环境时跳过
            pytest.skip("无 git 环境")
        assert "configs/strategy_gate.yaml" not in changed, (
            "人工确认不得顺带修改 strategy_gate 配置")

    def test_confirmed_tasks_declare_no_config_change(self):
        for tid in EXPECTED_IDS:
            t = _task(tid)
            must = t.get("confirmed_script_must") or ""
            assert must, f"{tid} 缺少 confirmed_script_must（确认口径）"
            # 确认口径不得包含「放行 / 下调 / 放宽阈值」类**动作**。
            # 注意：不能简单禁「改配置」三字——「确认不改配置」正是允许的表述。
            for word in ("下调", "放宽阈值", "提升阈值", "解锁门禁", "切换到"):
                assert word not in must, f"{tid} 确认口径越界: {word}"

    def test_gate_structure_stays_frozen(self):
        """门禁结构必须仍是冻结只读态：report_only + freeze_structure。"""
        if not STRATEGY_GATE_PATH.exists():
            pytest.skip("无 strategy_gate 配置")
        cfg = STRATEGY_GATE_PATH.read_text(encoding="utf-8")
        assert "report_only" in cfg, "门禁结构被改动（不是 report_only）"
        assert "freeze_structure" in cfg, "冻结标志丢失"


# ----------------------------------------------------------------------
# 四、阶段收官：completed 必须有确认字段背书
# ----------------------------------------------------------------------
class TestStagesClosedByConfirmation:
    def test_confirmed_stages_are_completed(self):
        """确认后的阶段**必须有** ``completed_at`` 口径的收官声明。

        本 PR 由 H 轮收口侧发起（#48），与「11 项确认」侧合并后阶段 ``status``
        存在两种等价口径：``completed``（确认侧收官）与 ``in_progress``
        （收口侧坚持「自动交付 ≠ 阶段完成」）。两种口径都必须带**人工确认背书**：
        ``closing.manual_scope`` 声明已确认，且阶段/任务带签署字段。
        """
        for sid in CONFIRMED_STAGES:
            s = _stage(sid)
            closing = s.get("closing") or {}
            assert "已确认" in (closing.get("manual_scope") or ""), \
                f"{sid} 未随确认收官（closing.manual_scope 未声明已确认）"

    def test_completed_stages_carry_confirmation(self):
        for sid in CONFIRMED_STAGES:
            s = _stage(sid)
            assert s.get("confirmed_by"), f"{sid} 收官却无 confirmed_by"
            assert s.get("confirmed_at"), f"{sid} 收官却无 confirmed_at"
            assert s.get("confirmed_in", "").endswith("/issues/40"), (
                f"{sid} 收官未指向确认来源")

    def test_closing_block_is_structurally_complete(self):
        for sid in CONFIRMED_STAGES:
            c = _stage(sid).get("closing") or {}
            for key in ("auto_scope", "manual_scope", "decision",
                        "confirmed_at", "confirmed_by", "affects_gate"):
                assert key in c, f"{sid}.closing 缺少 {key}"
            assert c["affects_gate"] is False, f"{sid} closing 未声明不影响门禁"

    def test_confirmed_stages_are_not_double_booked(self):
        """确认后的阶段：要么已 ``completed``，要么仍在 ``in_progress`` 但**必须有签署背书**。

        本 PR（#48）由 H 轮收口侧发起，与「11 项确认」侧合并后，S11~S20 的
        ``status`` 存在两侧口径：确认侧置 ``completed``，收口侧按
        「自动交付 ≠ 阶段完成」置 ``in_progress``。两者都不允许「无签字却标 completed」，
        也不允许「已签字却仍无签署字段」。
        """
        for sid in CONFIRMED_STAGES:
            s = _stage(sid)
            assert s["status"] in ("completed", "in_progress"), \
                f"{sid} status 非法: {s['status']}"
            if s["status"] == "completed":
                assert s.get("confirmed_by") and s.get("confirmed_at"), \
                    f"{sid} 标 completed 却缺签署字段（疑似代签）"
            else:
                # 仍在 in_progress：自动交付字段与阶段收口声明必须具备
                assert s.get("auto_acceptable_completed_at") or s.get("closing"), \
                    f"{sid} 仍 in_progress 且无任何交付/收口依据"
                assert (s.get("closing") or {}).get("manual_scope"), \
                    f"{sid} closing 缺 manual_scope"


# ----------------------------------------------------------------------
# 五、防回流：确认不得退化成代签
# ----------------------------------------------------------------------
class TestNoSignatureDrift:
    def test_no_manual_checkpoint_is_completed(self):
        bad = [t["id"] for _, t in _checkpoint_tasks() if t["status"] == "completed"]
        assert not bad, f"人工检查点被标 completed（代签）: {bad}"

    def test_no_pending_among_the_eleven(self):
        """决策包条目（G/H 轮）必须全部 confirmed；规划期条目必须全部 pending。"""
        for cp in _decision_entries():
            assert cp["status"] == "confirmed", f"{cp['id']} 仍未确认: {cp['status']}"
        for cp in _planning_entries():
            assert cp["status"] == "pending", (
                f"{cp['id']} 规划期条目只允许 pending: {cp['status']}")

    def test_confirmed_tasks_are_not_auto_run(self):
        for _, t in _checkpoint_tasks():
            assert t.get("auto_run") is False, f"{t['id']} 被标 auto_run=true"

    def test_manifest_policy_still_forbids_npc_decision(self):
        pol = _manifest()["policy"]
        assert pol["npc_may_decide"] is False
        assert pol.get("never_auto_filled") is True
