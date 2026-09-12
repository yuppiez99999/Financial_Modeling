"""S16~S20 路线守卫：高质量项目集成 2 轮（Issue #40 · H1~H5）排期完整性。

设计要点：
  - 先排期后集成：本轮交付的**排期计划**必须自洽、可推进、可审计；
  - 边界优先：全部阶段 affects_gate 恒 false（不改门禁）、人工检查点不得自动通过；
  - 阶段结构同构：id/name/tasks/auto_acceptable/manual_checkpoint/source/depends_on 齐备；
  - 依赖闭环：depends_on 指向的阶段必须存在于 plan.json；
  - 淘汰清单：明确被否掉的候选不得以阶段形式回流。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = PROJECT_ROOT / "schedule" / "plan.json"
CANDIDATES_DOC = PROJECT_ROOT / "00_kickoff" / "high_value_projects_round2_candidates.md"
THIRD_PARTY = PROJECT_ROOT / "docs" / "THIRD_PARTY.md"

ROUND2_STAGE_IDS = ["S16", "S17", "S18", "S19", "S20"]


def _load_plan():
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def _round2_stages():
    plan = _load_plan()
    by_id = {s["id"]: s for s in plan["stages"]}
    return [by_id[sid] for sid in ROUND2_STAGE_IDS]


def test_round2_stages_exist():
    """H1~H5 五个阶段必须都已写入排期。"""
    ids = [s["id"] for s in _round2_stages()]
    assert ids == ROUND2_STAGE_IDS


def test_round2_stages_are_pending_not_fake_completed():
    """不得出现「已排期即已完成」的假进度。

    S16 已真实开工（T16.3 提前交付，2026-09-12，Issue #29）：开工须有
    started_at 证据；completed 任务须有 completed_at 与 result；
    人工检查点任务恒 pending（不可自动通过）。
    """
    for s in _round2_stages():
        if s["status"] != "pending":
            assert s.get("started_at"), f"{s['id']} 非 pending 却无 started_at"
        for t in s["tasks"]:
            if t["id"] in (s.get("manual_checkpoint") or []):
                assert t["status"] == "pending", \
                    f"{t['id']} 是人工检查点，不得自动完成"
            elif t["status"] == "completed":
                assert t.get("completed_at") and t.get("result"), \
                    f"{t['id']} 标 completed 却无 completed_at/result"


def test_round2_stages_have_required_fields():
    """阶段结构同构：缺任一字段视为排期不完整。"""
    required = ["id", "name", "status", "scheduled_dates", "tasks",
                "auto_acceptable", "manual_checkpoint", "source", "depends_on", "note"]
    for s in _round2_stages():
        for key in required:
            assert key in s, f"{s.get('id')} 缺少字段 {key}"
        assert s["tasks"], f"{s['id']} 必须至少有一个任务"
        assert s["source"], f"{s['id']} 必须标注候选来源"


def test_round2_task_ids_are_consistent():
    """任务 id 必须与阶段号自洽（T16.x 属 S16），防复制粘贴错位。"""
    for s in _round2_stages():
        num = s["id"][1:]
        for t in s["tasks"]:
            assert t["id"].startswith(f"T{num}."), f"{t['id']} 不属于阶段 {s['id']}"


def test_round2_dates_are_sequential_and_non_overlapping():
    """排期日期必须推进且不重叠：先 H1→H5 顺序交付。"""
    prev_end = None
    for s in _round2_stages():
        start, end = [d.strip() for d in s["scheduled_dates"].split("~")]
        assert start <= end, f"{s['id']} 起止日期颠倒"
        if prev_end is not None:
            assert start > prev_end, f"{s['id']} 与上一阶段日期重叠或倒序"
        prev_end = end


def test_round2_manual_checkpoints_not_in_auto_acceptable():
    """人工检查点绝不能被标成自动可接受（防选择自由度被自动吞掉）。"""
    for s in _round2_stages():
        overlap = set(s["auto_acceptable"]) & set(s["manual_checkpoint"])
        assert not overlap, f"{s['id']} 检查点与自动项冲突: {sorted(overlap)}"
        assert s["manual_checkpoint"], f"{s['id']} 本轮必须保留至少一个人工检查点"


def test_round2_depends_on_exists_and_acyclic():
    """依赖必须指向真实存在的阶段，且整体按 id 递增推进（无环）。"""
    plan = _load_plan()
    by_id = {s["id"]: s for s in plan["stages"]}
    order = {s["id"]: i for i, s in enumerate(plan["stages"])}
    for s in _round2_stages():
        for dep in s["depends_on"]:
            assert dep in by_id, f"{s['id']} 依赖不存在的阶段 {dep}"
            assert order[dep] < order[s["id"]], f"{s['id']} 依赖了更晚的阶段 {dep}（成环）"


def test_round2_affects_gate_is_false_by_construction():
    """本轮零门禁改动：阶段与任务描述中不得出现「降低门槛/放宽门禁」类表述。"""
    forbidden = ["降低门禁", "放宽门禁", "下调门禁", "把 52% 线改低", "改低门禁"]
    for s in _round2_stages():
        blob = s["name"] + s.get("note", "") + " ".join(t["name"] for t in s["tasks"])
        for word in forbidden:
            assert word not in blob, f"{s['id']} 出现门禁放宽表述: {word}"


def test_candidate_doc_and_third_party_registered():
    """选型依据与第三方登记必须落盘：排期不能只有结论没有依据。"""
    assert CANDIDATES_DOC.exists(), "缺少候选评估矩阵文档"
    doc = CANDIDATES_DOC.read_text(encoding="utf-8")
    for sid in ["H1", "H2", "H3", "H4", "H5"]:
        assert sid in doc, f"候选文档缺少 {sid}"
    assert "MAPIE" in doc and "hmmlearn" in doc, "候选文档缺少入选项目明细"

    tp = THIRD_PARTY.read_text(encoding="utf-8")
    assert "H1~H5" in tp, "第三方登记缺少 H1~H5 轮次条目"
    assert "规划期登记" in tp or "规划期" in tp, "第三方登记必须声明规划期状态（未实际引入）"


def test_rejected_candidates_not_reintroduced_as_stages():
    """明确淘汰的候选，不得以阶段名形式悄悄回流（避免无效方向重复投入）。"""
    rejected_markers = ["tsfresh", "rqalpha", "hikyuu", "FinGPT", "FinRobot", "backtesting.py"]
    plan = _load_plan()
    stage_blob = " ".join(s["name"] for s in plan["stages"] if s["id"] in ROUND2_STAGE_IDS)
    for marker in rejected_markers:
        assert marker not in stage_blob, f"淘汰候选 {marker} 疑似回流进排期"


def test_round2_dates_follow_previous_round():
    """H 轮排期必须接在既有 G 轮（S15）之后，不留空窗也不倒挂。"""
    plan = _load_plan()
    by_id = {s["id"]: s for s in plan["stages"]}
    g5_end = by_id["S15"]["scheduled_dates"].split("~")[1].strip()
    h1_start = by_id["S16"]["scheduled_dates"].split("~")[0].strip()
    assert h1_start > g5_end, f"H1 起始 {h1_start} 应晚于 G5 结束 {g5_end}"
