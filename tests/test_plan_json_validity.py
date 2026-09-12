"""计划推进守卫：plan.json 必须可解析，且 current_stage_index 不越界。

背景：`schedule/run_daily.py` 是无人值守的每日自动推进脚本。一旦 `plan.json`
被写成非法 JSON（例如 `ConvertTo-Json -Depth 6` 截断），后续每日任务会直接崩掉，
且没有任何告警——排期静默停摆。本测试把该约束固化为可执行断言。
"""
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = PROJECT_ROOT / "schedule" / "plan.json"


def _load_plan():
    assert PLAN_PATH.exists(), f"缺少排期文件 {PLAN_PATH}"
    try:
        return json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:  # pragma: no cover - 失败即缺陷
        pytest.fail(f"schedule/plan.json 不是合法 JSON（每日推进将整体崩塌）: {e}")


def test_plan_json_is_parseable():
    plan = _load_plan()
    assert isinstance(plan, dict)
    assert plan.get("stages"), "plan.json 必须包含 stages"


def test_plan_stage_index_in_range():
    plan = _load_plan()
    stages = plan["stages"]
    idx = int(plan.get("current_stage_index", 0))
    assert 0 <= idx < len(stages), (
        f"current_stage_index={idx} 越界（stages 长度 {len(stages)}），"
        "run_daily.py 会因 stages[idx] 抛 IndexError"
    )


def test_plan_task_index_in_range_for_current_stage():
    plan = _load_plan()
    idx = int(plan.get("current_stage_index", 0))
    tasks = plan["stages"][idx].get("tasks", [])
    tidx = int(plan.get("current_task_index", 0))
    assert tasks, "当前阶段必须至少有一个任务"
    assert 0 <= tidx < len(tasks), (
        f"current_task_index={tidx} 越界（当前阶段任务数 {len(tasks)}）"
    )


# 阶段/任务的「已落定」状态集合。
# 背景（2026-09-12）：人工检查点由**用户确认**后不再是 `pending`，而是
# `confirmed`（带 confirmed_by / confirmed_at / decision）。`confirmed` 与
# `completed` 同属「已落定」——但两者的**来源不同**：`completed` 来自自动任务，
# `confirmed` **只能来自人工签字**（守卫见 test_roadmap_manual_checkpoints.py）。
DONE_STATUSES = {"completed", "confirmed"}


def test_plan_tasks_are_completed_atomically_per_stage():
    """阶段标记 completed 的前提是其下任务全部已落定（防「阶段假完成」）。

    已落定 = `completed`（自动任务）或 `confirmed`（人工签字确认）。
    任何仍为 `pending` 的任务都会让本断言失败。
    """
    plan = _load_plan()
    for stage in plan["stages"]:
        if stage.get("status") == "completed":
            pending = [t["id"] for t in stage.get("tasks", [])
                       if t.get("status") not in DONE_STATUSES]
            assert not pending, f"阶段 {stage.get('id')} 标记完成但仍有未完成任务: {pending}"


def test_plan_confirmed_tasks_carry_human_signature():
    """`confirmed` 是人工签字态：必须带签署人与时间（防自动流程自签）。"""
    plan = _load_plan()
    for stage in plan["stages"]:
        for t in stage.get("tasks", []):
            if t.get("status") != "confirmed":
                continue
            assert t.get("confirmed_by"), f"{t['id']} 标 confirmed 却无 confirmed_by"
            assert t.get("confirmed_at"), f"{t['id']} 标 confirmed 却无 confirmed_at"
            assert t.get("decision") in {"keep", "defer", "reject", "cancel",
                                         "moot_by_convention"}, \
                f"{t['id']} 决策词表外的取值: {t.get('decision')}"


def test_blockers_is_a_list():
    plan = _load_plan()
    assert isinstance(plan.get("blockers", []), list)
