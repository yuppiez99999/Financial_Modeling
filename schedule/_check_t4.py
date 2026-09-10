# _check_t4.py
from pathlib import Path
import json

p = Path(__file__).resolve().parent.parent / "schedule" / "plan.json"
plan = json.loads(p.read_text(encoding="utf-8"))
stage = plan["stages"][3]
print("STAGE_NAME=", stage["name"])
print("STAGE_STATUS=", stage["status"])
for t in stage["tasks"]:
    print(t["id"], t["status"])
