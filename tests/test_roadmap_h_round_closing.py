"""H 轮（S16~S20）收口守卫：自动任务收官 ≠ 已决策（fail-close）。

设计要点（这是**纪律测试**，不是功能测试）：
  - **自动任务收官**必须有可审计证据：`auto_acceptable` 全 completed、
    每个任务带 `completed_at` + `result`；
  - **阶段不得提前 completed**：人工检查点签字前 `status` 只能是
    `pending` / `in_progress`；
  - **人工检查点恒 pending**：H 轮 5 项不得被自动流程代签；
  - **决策包必须覆盖 H 轮全部检查点**：T16.4 / T17.4 / T18.4 / T19.4 / T20.4
    必须在 `schedule/manual_checkpoints.json` 且 priority 唯一；
  - **结论文档必须存在**：H4 / H5 结论文件不得只在清单里挂空路径；
  - **边界恒定**：H 轮不得出现门禁放宽表述，`affects_gate` 相关字段恒 false。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = PROJECT_ROOT / "schedule" / "plan.json"
MANIFEST_PATH = PROJECT_ROOT / "schedule" / "manual_checkpoints.json"

H_ROUND_STAGE_IDS = ["S16", "S17", "S18", "S19", "S20"]
H_ROUND_CHECKPOINT_IDS = {"T16.4", "T17.4", "T18.4", "T19.4", "T20.4"}
H_ROUND_CONCLUSION_DOCS = [
    "00_kickoff/conformal_interval_conclusion.md",
    "00_kickoff/s17_overfit_audit_conclusion.md",
    "00_kickoff/s18_regime_conclusion.md",
    "00_kickoff/probability_calibration_conclusion.md",
    "00_kickoff/research_assist_conclusion.md",
]
CLOSING_DOC = PROJECT_ROOT / "00_kickoff" / "manual_checkpoints_round_g_h.md"


def _plan() -> dict:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _h_stages() -> list[dict]:
    by_id = {s["id"]: s for s in _plan()["stages"]}
    return [by_id[sid] for sid in H_ROUND_STAGE_IDS]


# ----------------------------------------------------------------------
# 自动任务收官：可审计
# ----------------------------------------------------------------------
class TestAutoScopeClosed:
    def test_all_auto_acceptable_tasks_completed(self):
        """H 轮 auto_acceptable 必须全部 completed，否则不算收官。"""
        for s in _h_stages():
            for tid in s["auto_acceptable"]:
                task = next(t for t in s["tasks"] if t["id"] == tid)
                assert task["status"] == "completed", f"{tid} 未完成，S{ s['id'] } 不得称收官"

    def test_completed_tasks_have_audit_fields(self):
        """标 completed 的任务必须有 completed_at + result（可审计）。"""
        for s in _h_stages():
            for t in s["tasks"]:
                if t["status"] == "completed":
                    assert t.get("completed_at"), f"{t['id']} 缺 completed_at"
                    assert t.get("result"), f"{t['id']} 缺 result"

    def test_closing_fields_present_and_false_gate(self):
        """阶段收口字段齐备，且 closing.affects_gate 恒 false。"""
        required = ["round", "auto_acceptable_completed_at",
                    "auto_acceptable_verified", "closing"]
        for s in _h_stages():
            for key in required:
                assert key in s, f"{s['id']} 缺收口字段 {key}"
            c = s["closing"]
            for key in ("auto_scope", "manual_scope", "evidence",
                        "checkpoint_manifest", "verified_at", "affects_gate"):
                assert key in c, f"{s['id']}.closing 缺 {key}"
            assert c["affects_gate"] is False, f"{s['id']} closing 声称影响到门禁"

    def test_round_tag_matches_stage(self):
        """round 标记必须与阶段号自洽（S16→H1 … S20→H5）。"""
        for i, s in enumerate(_h_stages(), start=1):
            assert s["round"] == f"H{i}", f"{s['id']} round 标记错位: {s['round']}"


# ----------------------------------------------------------------------
# 阶段不得提前 completed
# ----------------------------------------------------------------------
class TestStageNotPrematurelyCompleted:
    def test_stage_status_not_completed(self):
        """人工检查点未签字 → 阶段不得 completed。"""
        for s in _h_stages():
            assert s["status"] in ("pending", "in_progress"), (
                f"{s['id']} 在人工检查点未签字时标 {s['status']}（假进度）")
            assert s["status"] != "completed", f"{s['id']} 不得 completed"

    def test_manual_checkpoints_stay_pending(self):
        """H 轮 5 项人工检查点恒 pending，且无 completed_at / result。"""
        for s in _h_stages():
            manual = set(s["manual_checkpoint"])
            for t in s["tasks"]:
                if t["id"] in manual:
                    assert t["status"] == "pending", f"{t['id']} 被代签"
                    assert t.get("auto_run") is False, f"{t['id']} 标 auto_run"
                    assert not t.get("completed_at"), f"{t['id']} 有 completed_at"
                    assert not t.get("result"), f"{t['id']} 有 result"


# ----------------------------------------------------------------------
# 决策包覆盖 H 轮
# ----------------------------------------------------------------------
class TestManifestCoversHRound:
    def test_all_h_checkpoints_in_manifest(self):
        ids = {c["id"] for c in _manifest()["checkpoints"]}
        missing = H_ROUND_CHECKPOINT_IDS - ids
        assert not missing, f"决策包缺 H 轮检查点: {sorted(missing)}"

    def test_h_checkpoints_all_pending(self):
        for cp in _manifest()["checkpoints"]:
            if cp["id"] in H_ROUND_CHECKPOINT_IDS:
                assert cp["status"] == "pending", f"{cp['id']} 状态被改动"

    def test_manifest_priorities_unique_and_sequential(self):
        ps = [c["priority"] for c in _manifest()["checkpoints"]]
        assert len(ps) == len(set(ps)), "决策包 priority 重复"
        assert sorted(ps) == list(range(1, len(ps) + 1)), "priority 必须 1..N 连续"

    def test_h_checkpoints_have_evidence_and_doc(self):
        for cp in _manifest()["checkpoints"]:
            if cp["id"] in H_ROUND_CHECKPOINT_IDS:
                assert cp.get("evidence"), f"{cp['id']} 缺 evidence"
                assert cp.get("doc"), f"{cp['id']} 缺 doc"
                assert cp.get("round"), f"{cp['id']} 缺 round 标记"

    def test_manifest_declares_rounds_covered(self):
        rc = _manifest().get("rounds_covered") or []
        assert any("H 轮" in r for r in rc), "决策包须声明覆盖 H 轮"


# ----------------------------------------------------------------------
# 结论文档存在且自述边界
# ----------------------------------------------------------------------
class TestConclusionDocs:
    @pytest.mark.parametrize("rel", H_ROUND_CONCLUSION_DOCS)
    def test_doc_exists_and_states_boundary(self, rel: str):
        path = PROJECT_ROOT / rel
        assert path.exists(), f"缺结论文档 {rel}"
        doc = path.read_text(encoding="utf-8")
        assert "affects_gate" in doc, f"{rel} 未声明 affects_gate 纪律"
        assert "不代签" in doc, f"{rel} 未声明不代签"

    def test_closing_doc_covers_all_h_checkpoints(self):
        assert CLOSING_DOC.exists(), "缺 H 轮收口决策包文档"
        doc = CLOSING_DOC.read_text(encoding="utf-8")
        for cid in H_ROUND_CHECKPOINT_IDS:
            assert cid in doc, f"收口文档未覆盖 {cid}"


# ----------------------------------------------------------------------
# 边界恒定
# ----------------------------------------------------------------------
class TestBoundaryHolds:
    def test_no_gate_relaxation_wording(self):
        forbidden = ["降低门禁", "放宽门禁", "下调门禁", "改低门禁", "降低门槛"]
        for s in _h_stages():
            blob = s["name"] + (s.get("note") or "")
            blob += " ".join(t["name"] for t in s["tasks"])
            blob += json.dumps(s.get("closing") or {}, ensure_ascii=False)
            for word in forbidden:
                assert word not in blob, f"{s['id']} 出现门禁放宽表述: {word}"

    def test_auto_acceptable_never_contains_manual_checkpoint(self):
        for s in _h_stages():
            overlap = set(s["auto_acceptable"]) & set(s["manual_checkpoint"])
            assert not overlap, f"{s['id']} 检查点混入自动项: {sorted(overlap)}"

    def test_manual_scope_names_the_pending_checkpoints(self):
        """closing.manual_scope 必须点名未决策的检查点，不能含糊。"""
        for s in _h_stages():
            scope = s["closing"]["manual_scope"]
            for cid in s["manual_checkpoint"]:
                assert cid in scope, f"{s['id']}.closing.manual_scope 未点名 {cid}"

    def test_evidence_paths_are_repo_relative(self):
        for s in _h_stages():
            ev = s["closing"]["evidence"]
            assert not ev.startswith("/"), f"{s['id']} evidence 不得是绝对路径"
            assert "/" in ev or ev.endswith(".json"), f"{s['id']} evidence 形态可疑: {ev}"
