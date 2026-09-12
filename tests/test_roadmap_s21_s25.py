"""S21~S25 路线守卫：高质量项目集成 3 轮（Issue #54 · I1~I5）排期完整性。

设计要点（与 H 轮守卫同构，见 tests/test_roadmap_s16_s20.py）：
  - 先排期后集成：本轮交付的**排期计划**必须自洽、可推进、可审计；
  - 边界优先：全部阶段 affects_gate 恒 false（不改门禁）、人工检查点不得自动通过；
  - 阶段结构同构：id/name/tasks/auto_acceptable/manual_checkpoint/source/depends_on 齐备；
  - 依赖闭环：depends_on 指向的阶段必须存在于 plan.json 且顺序在前（无环）；
  - 淘汰清单：明确被否掉的候选（skfolio/pypbo/PyPortfolioOpt 运行时等）不得以阶段形式回流。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = PROJECT_ROOT / "schedule" / "plan.json"
CANDIDATES_DOC = PROJECT_ROOT / "00_kickoff" / "high_value_projects_round3_candidates.md"

ROUND3_STAGE_IDS = ["S21", "S22", "S23", "S24", "S25"]
ROUND3_ROUNDS = ["I1", "I2", "I3", "I4", "I5"]


def _load_plan():
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def _round3_stages():
    plan = _load_plan()
    by_id = {s["id"]: s for s in plan["stages"]}
    return [by_id[sid] for sid in ROUND3_STAGE_IDS]


def test_round3_stages_exist():
    """I1~I5 五个阶段必须都已写入排期，且置于 H 轮（S16~S20）之后。"""
    plan = _load_plan()
    ids = [s["id"] for s in plan["stages"]]
    for sid in ROUND3_STAGE_IDS:
        assert sid in ids, f"缺少阶段 {sid}"
    assert ids[-5:] == ROUND3_STAGE_IDS, "S21~S25 必须是排期最后五个阶段"


def test_round3_progress_is_honest_no_fake_completion():
    """进度必须诚实（与 H 轮守卫同构）：
      - 阶段允许 pending / in_progress / completed，但 completed 必须建立在
        「全部任务已落定 + 检查点 confirmed + 阶段带签署字段」之上；
      - 非 pending 阶段必须有 started_at 证据；
      - 标 completed 的任务必须有 completed_at 与 result；
      - 人工检查点只允许 pending / confirmed，绝不允许 completed（防代签）。
    """
    for s in _round3_stages():
        assert s["status"] in ("pending", "in_progress", "completed"), \
            f"{s['id']} 阶段状态非法: {s['status']}"
        if s["status"] != "pending":
            assert s.get("started_at"), f"{s['id']} 非 pending 却无 started_at"
        manual = set(s.get("manual_checkpoint") or [])
        for t in s["tasks"]:
            assert t["status"] in ("pending", "confirmed", "completed"), \
                f"{t['id']} 状态非法"
            if t["id"] in manual:
                assert t["status"] != "completed", \
                    f"人工检查点 {t['id']} 不得被自动流程标记完成（代签）"
                if t["status"] == "confirmed":
                    assert t.get("confirmed_by") and t.get("confirmed_at"), \
                        f"{t['id']} 标 confirmed 却无人工签署字段"
            elif t["status"] == "completed":
                assert t.get("completed_at") and t.get("result"), \
                    f"{t['id']} 标 completed 却无 completed_at/result"
        if s["status"] == "completed":
            assert s.get("confirmed_by") and s.get("confirmed_at"), \
                f"{s['id']} 标 completed 却无人工确认字段（防阶段假完成）"
            assert (s.get("closing") or {}).get("affects_gate") is False, \
                f"{s['id']} closing 未声明 affects_gate=false"


def test_round3_stages_have_required_fields():
    """阶段结构同构：缺任一字段视为排期不完整。"""
    required = ["id", "name", "status", "scheduled_dates", "tasks",
                "auto_acceptable", "manual_checkpoint", "source", "depends_on",
                "note", "round", "affects_gate"]
    for s in _round3_stages():
        for key in required:
            assert key in s, f"{s.get('id')} 缺少字段 {key}"
        assert s["tasks"], f"{s['id']} 必须至少有一个任务"
        assert s["source"].startswith("Issue #54"), f"{s['id']} 来源必须标注 Issue #54"
        assert s["affects_gate"] is False, f"{s['id']} affects_gate 必须为 false"
        assert s["round"] in ROUND3_ROUNDS, f"{s['id']} round 字段非法: {s.get('round')}"


def test_round3_task_ids_are_consistent():
    """任务 id 必须与阶段号自洽（T21.x 属 S21），防复制粘贴错位。"""
    for s in _round3_stages():
        num = s["id"][1:]
        for t in s["tasks"]:
            assert t["id"].startswith(f"T{num}."), f"{t['id']} 不属于阶段 {s['id']}"


def test_round3_dates_are_sequential_and_non_overlapping():
    """排期日期必须推进且不重叠：先 I1→I5 顺序交付。"""
    prev_end = None
    for s in _round3_stages():
        start, end = [d.strip() for d in s["scheduled_dates"].split("~")]
        assert start <= end, f"{s['id']} 起止日期颠倒"
        if prev_end is not None:
            assert start > prev_end, f"{s['id']} 与上一阶段日期重叠或倒序"
        prev_end = end


def test_round3_manual_checkpoints_not_in_auto_acceptable():
    """人工检查点绝不能被标成自动可接受（防选择自由度被自动吞掉）。"""
    for s in _round3_stages():
        overlap = set(s["auto_acceptable"]) & set(s["manual_checkpoint"])
        assert not overlap, f"{s['id']} 检查点与自动项冲突: {sorted(overlap)}"
        assert s["manual_checkpoint"], f"{s['id']} 本轮必须保留至少一个人工检查点"
        for t in s["tasks"]:
            if t["id"] in s["manual_checkpoint"]:
                assert t["auto_run"] is False, f"{t['id']} 人工检查点不得 auto_run"


def test_round3_depends_on_exists_and_acyclic():
    """依赖必须指向真实存在的阶段，且整体按 id 递增推进（无环）。"""
    plan = _load_plan()
    by_id = {s["id"]: s for s in plan["stages"]}
    order = {s["id"]: i for i, s in enumerate(plan["stages"])}
    for s in _round3_stages():
        for dep in s["depends_on"]:
            assert dep in by_id, f"{s['id']} 依赖不存在的阶段 {dep}"
            assert order[dep] < order[s["id"]], f"{s['id']} 依赖了更晚的阶段 {dep}（成环）"


def test_round3_affects_gate_is_false_by_construction():
    """本轮零门禁改动：阶段与任务描述中不得出现「降低门槛/放宽门禁」类表述。"""
    forbidden = ["降低门禁", "放宽门禁", "下调门禁", "把 52% 线改低", "改低门禁"]
    for s in _round3_stages():
        blob = s["name"] + s.get("note", "") + " ".join(t["name"] for t in s["tasks"])
        for word in forbidden:
            assert word not in blob, f"{s['id']} 出现门禁放宽表述: {word}"


def test_round3_rejected_candidates_do_not_leak_back_in():
    """淘汰清单中的候选不得以「引入运行时」的形式回流排期。"""
    doc = CANDIDATES_DOC.read_text(encoding="utf-8")
    for sid in ROUND3_ROUNDS:
        assert sid in doc, f"候选矩阵缺少 {sid} 的选型记录"
    # 被淘汰者不得出现在任何阶段的任务描述里（作为「引入」对象）
    banned = ["skfolio", "pypbo", "riskfolio", "mlflow"]
    for s in _round3_stages():
        blob = s["name"] + " ".join(t["name"] for t in s["tasks"])
        for name in banned:
            assert name not in blob.lower(), (
                f"{s['id']} 把淘汰候选 {name} 写回了任务（只允许留在淘汰清单）")


def test_round3_zero_new_runtime_deps_in_planning():
    """规划期不得新增运行时依赖：requirements.txt 保持缺省不装。"""
    req = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    for pkg in ("evidently", "pypfopt", "cvxpy"):
        for line in req.splitlines():
            stripped = line.strip()
            if stripped.startswith(pkg):
                pytest.fail(
                    f"规划期不得解注释 {pkg}（候选矩阵第五节：准入评估通过后才登记）")
