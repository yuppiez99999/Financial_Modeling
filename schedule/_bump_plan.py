# _bump_plan.py
from pathlib import Path
import json
from datetime import datetime, timezone

root = Path(__file__).resolve().parent.parent
plan_path = root / "schedule" / "plan.json"
plan = json.loads(plan_path.read_text(encoding="utf-8"))

now_iso = lambda: datetime.now(timezone.utc).isoformat()

idx = int(plan.get("current_stage_index", 0))
stages = plan.get("stages", [])
stage = stages[idx]
tasks = stage.get("tasks", [])

def complete_task(task_id):
    for t in tasks:
        if t.get("id") == task_id and t.get("status") != "completed":
            t["status"] = "completed"
            t["completed_at"] = now_iso()

# 本次开发完成项
complete_task("T1.1")
complete_task("T1.2")
complete_task("T1.3")

# 阶段完成判断
stage_done = all(t.get("status") == "completed" for t in tasks)
if stage_done:
    stage["status"] = "completed"
    stage["completed_at"] = now_iso()
    if idx < len(stages) - 1:
        plan["current_stage_index"] = idx + 1
        plan["current_task_index"] = 0
    else:
        plan["current_stage_index"] = idx
        plan["current_task_index"] = len(tasks) - 1

plan["last_run_date"] = datetime.now().strftime("%Y-%m-%d")
plan["blockers"] = list(dict.fromkeys(plan.get("blockers", [])))

plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
print("bumped")
print("stage_status=", stage.get("status"))
print("next_stage_index=", plan.get("current_stage_index"))
print("next_task_index=", plan.get("current_task_index"))
