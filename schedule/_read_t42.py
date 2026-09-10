# _read_t42.py
from pathlib import Path
import json

p = Path(__file__).resolve().parent.parent / "schedule" / "plan.json"
plan = json.loads(p.read_text(encoding="utf-8"))
stage = plan["stages"][3]
for t in stage["tasks"]:
    if t["id"] == "T4.2":
        print("TASK_ID=", t["id"])
        print("TASK_NAME=", t["name"])
        print("DESC=", t.get("desc"))
        print("ACCEPTANCE=", t.get("acceptance"))
        print("AUTO_RUN=", t.get("auto_run"))
        break
