# run_daily.py
import json
import os
import datetime
import subprocess
import sys
from pathlib import Path

def load_plan(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_plan(path, plan):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

def today_iso():
    return datetime.date.today().isoformat()

def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def path_exists(p):
    return Path(p).exists()

def git_clone_shallow(dest, url):
    if path_exists(dest):
        return True
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, dest],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False

def smoke_ok(proj, libname, url, must_have=None):
    dest = Path(proj) / "libs" / libname
    if not git_clone_shallow(str(dest), url):
        return False
    if must_have:
        return any((dest / x).exists() for x in must_have)
    return True

def run_day(plan_path, log_dir):
    plan = load_plan(plan_path)
    log_file = Path(log_dir) / f"run_{today_iso()}.json"

    entry = {
        "run_date": today_iso(),
        "ran_at": now_iso(),
        "stage": plan.get("current_stage_index"),
        "task": plan.get("current_task_index"),
        "actions": [],
        "result": "unknown",
        "blockers": [],
    }

    try:
        stages = plan.get("stages", [])
        idx = plan.get("current_stage_index", 0)
        tidx = plan.get("current_task_index", 0)

        if idx >= len(stages):
            entry["result"] = "done_all_stages"
            entry["actions"].append("所有阶段已完成")
            plan["result"] = entry["result"]
            plan["last_run_date"] = today_iso()
            plan["next_run_date"] = None
            save_plan(plan_path, plan)
            return

        stage = stages[idx]

        if stage.get("status") == "completed":
            next_idx = None
            for i, s in enumerate(stages):
                if s.get("status") == "pending":
                    next_idx = i
                    break
            if next_idx is None:
                entry["result"] = "done_all_stages"
                entry["actions"].append("所有阶段已完成")
                plan["result"] = entry["result"]
                plan["last_run_date"] = today_iso()
                plan["next_run_date"] = None
                save_plan(plan_path, plan)
                return
            plan["current_stage_index"] = next_idx
            plan["current_task_index"] = 0
            idx = next_idx
            stage = stages[idx]

        if stage.get("status") == "pending":
            stage["status"] = "in_progress"
            stage["started_at"] = now_iso()
            entry["actions"].append(f"阶段开始：{stage.get('name')}")

        task = stage["tasks"][tidx]
        task_id = task["id"]
        task_name = task["name"]
        auto_run = bool(task.get("auto_run", False))

        entry["actions"].append(
            f"准备推进：[{stage.get('name')}] {task_id} - {task_name} (auto_run={auto_run})"
        )

        proj = str(Path(plan_path).parent.parent)

        ok = False
        msg = ""

        if auto_run:
            if task_id == "T1.1":
                ok = smoke_ok(proj, "vectorbt", "https://github.com/vectorbt/vectorbt",
                              must_have=["README.md", "examples"])
                msg = "vectorbt 最小示例尝试完成" if ok else "vectorbt 最小示例尝试失败/未达标"
            elif task_id == "T1.3":
                ok = path_exists(Path(proj) / "backtest" / "run_minimal.py")
                msg = "最小回测脚本已存在" if ok else "最小回测脚本尚未创建"
            elif task_id == "T2.3":
                ok = smoke_ok(proj, "tsai", "https://github.com/timeseriesAI/tsai",
                              must_have=["README.md", "notebooks"])
                msg = "tsai 示例尝试完成" if ok else "tsai 示例尝试失败/未达标"
            elif task_id == "T2.4":
                ok = path_exists(Path(proj) / "models" / "tsai_baselines")
                msg = "tsai baseline 目录已存在" if ok else "tsai baseline 目录尚未创建"
            elif task_id == "T3.3":
                ok = path_exists(Path(proj) / "pipeline" / "mlfinlab" / "experiments")
                msg = "mlfinlab 试跑目录已存在" if ok else "mlfinlab 试跑目录尚未创建"
            elif task_id == "T5.1":
                ok = smoke_ok(proj, "FinRL", "https://github.com/AI4Finance-Foundation/FinRL",
                              must_have=["README.md", "examples"])
                msg = "FinRL 示例尝试完成" if ok else "FinRL 示例尝试失败/未达标"
            elif task_id == "T5.4":
                ok = path_exists(Path(proj) / "rl" / "finrl_experiments" / "results_and_decision.md")
                msg = "FinRL 结论文件已存在" if ok else "FinRL 结论文件尚未创建"
            else:
                msg = f"未知可自动任务 {task_id}"

            entry["actions"].append(f"自动任务结果：{msg}")

            if ok:
                task["status"] = "completed"
                task["completed_at"] = now_iso()
                entry["actions"].append(f"标记完成：{task_id}")
            else:
                entry["blockers"].append(f"{task_id} - {msg}")
                if msg not in plan.get("blockers", []):
                    plan.setdefault("blockers", []).append(msg)
        else:
            entry["actions"].append(f"人工检查点，跳过自动完成：{task_id}")

        stage_done = all(t.get("status") == "completed" for t in stage["tasks"])
        if stage_done:
            stage["status"] = "completed"
            stage["completed_at"] = now_iso()
            entry["actions"].append(f"阶段完成：{stage.get('name')}")
            if idx < len(stages) - 1:
                plan["current_stage_index"] = idx + 1
                plan["current_task_index"] = 0
            else:
                entry["result"] = "all_stages_completed"

        plan["last_run_date"] = today_iso()
        plan["next_run_date"] = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
        plan["result"] = entry["result"]
        plan["blockers"] = list(dict.fromkeys(plan.get("blockers", [])))

        save_plan(plan_path, plan)
    except Exception as e:  # noqa: BLE001 — 计划推进失败不得中断每日任务
        entry["result"] = "error"
        entry["actions"].append(f"异常：{e}")
        entry["blockers"].append(f"runtime_error: {e}")
        plan.setdefault("blockers", []).append(f"runtime_error: {e}")
        try:
            save_plan(plan_path, plan)
        except Exception:  # noqa: BLE001
            pass
    finally:
        try:
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)
        except Exception as e:  # noqa: BLE001
            print(f"[daily] 日志写入失败：{e}")

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    plan_path = root / "plan.json"
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_day(str(plan_path), str(log_dir))

