# _read_plan.py
from pathlib import Path
import json

root = Path(__file__).resolve().parent.parent
plan_path = root / "schedule" / "plan.json"
plan = json.loads(plan_path.read_text(encoding="utf-8"))

idx = int(plan.get("current_stage_index", 0))
tidx = int(plan.get("current_task_index", 0))
stages = plan.get("stages", [])
stage = stages[idx] if idx < len(stages) else None
task = stage["tasks"][tidx] if stage and tidx < len(stage.get("tasks", [])) else None

print("CURRENT_STAGE_INDEX=", idx)
print("CURRENT_TASK_INDEX=", tidx)
print("STAGE_NAME=", (stage or {}).get("name"))
print("STAGE_STATUS=", (stage or {}).get("status"))
print("TASK_ID=", (task or {}).get("id"))
print("TASK_NAME=", (task or {}).get("name"))
print("AUTO_RUN=", bool((task or {}).get("auto_run")))
