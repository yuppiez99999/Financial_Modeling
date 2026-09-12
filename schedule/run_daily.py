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

# 「已落定」状态集合：completed 来自自动任务，confirmed 来自人工签字。
# 背景（真实缺陷）：本脚本原先只认 "completed"，而人工检查点在用户确认后
# 的终态是 "confirmed"（见 schedule/manual_checkpoints.json 与
# tests/test_roadmap_manual_checkpoints.py）。于是：
#   - 阶段永远无法判定为完成（人工检查点恒不等于 completed）→ 排期停摆；
#   - 反过来，一旦某阶段被标 completed，脚本的推进分支只搜索 status=="pending"
#     的阶段、且**不回头检查已跳过阶段里未落定的任务** → 未完成任务被静默吞掉。
# 两者方向相反，都会让「自动部分交付到哪」无法审计。这里统一口径。
DONE_STATUSES = {"completed", "confirmed"}


def task_is_done(task):
    """任务是否已落定（自动完成或人工签字确认）。"""
    return str(task.get("status")) in DONE_STATUSES


def stage_is_done(stage):
    """阶段是否已落定：其下**全部**任务已落定，且阶段不得为空。"""
    tasks = stage.get("tasks") or []
    if not tasks:
        return False
    return all(task_is_done(t) for t in tasks)


def _block_and_stop(plan, plan_path, entry, result, msg):
    """把「拒绝推进」的坏状态如实落盘并停止本轮（三处守卫共用）。

    统一出口的意义：**拒绝推进的原因必须可审计**。原先各分支用
    `entry["blockers"]` 零散记录、且异常时被 except 吞成 runtime_error，
    排期停摆却看不出原因。这里保证 result / blockers / next_run_date
    三处口径一致，且不再排下一次自动推进（避免无人值守时空转）。
    """
    entry["result"] = result
    entry["actions"].append(msg)
    entry["blockers"].append(msg)
    if msg not in plan.get("blockers", []):
        plan.setdefault("blockers", []).append(msg)
    plan["result"] = result
    plan["last_run_date"] = today_iso()
    plan["next_run_date"] = None
    save_plan(plan_path, plan)


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
            # 守卫（真实缺陷）：阶段被标 completed，但其下仍有未落定任务时，
            # 绝不能直接跳到下一阶段 —— 那等于把未完成项静默吞掉、并让
            # plan.json 对外宣称"已交付"。此处改为如实记录阻塞并停在本阶段。
            unfinished = [t.get("id") for t in (stage.get("tasks") or [])
                          if not task_is_done(t)]
            if unfinished:
                msg = (f"阶段 {stage.get('id')} 标 completed 但仍有未落定任务："
                       f"{unfinished}（拒绝跳过，防止假进度）")
                _block_and_stop(plan, plan_path, entry,
                                "blocked_inconsistent_stage", msg)
                return

            next_idx = None
            for i, s in enumerate(stages):
                # 跳过未落定条件与上面同源：既看 pending，也看任何未落定状态
                if s.get("status") in ("pending", "in_progress") and not stage_is_done(s):
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
            entry["actions"].append(f"阶段开始：{stage.get('name') or stage.get('id')}")

        # 任务定位（真实缺陷）：原实现用 stage["tasks"][tidx] / task["name"]
        # 直接取值，一旦 current_task_index 越界或任务缺 name 字段，整轮推进
        # 会抛异常并被 except 吞成 "runtime_error" —— 排期静默停摆且看不出原因。
        # 这里显式校验索引与缺省取值，把"坏数据"变成**可见的阻塞**。
        stage_tasks = stage.get("tasks") or []
        if not stage_tasks:
            msg = f"阶段 {stage.get('id')} 没有任何任务，拒绝判定为完成（防假进度）"
            _block_and_stop(plan, plan_path, entry, "blocked_empty_stage", msg)
            return
        if not 0 <= tidx < len(stage_tasks):
            msg = (f"阶段 {stage.get('id')} 的 current_task_index={tidx} 越界"
                   f"（任务数 {len(stage_tasks)}）")
            _block_and_stop(plan, plan_path, entry,
                            "blocked_task_index_out_of_range", msg)
            return

        task = stage_tasks[tidx]
        task_id = task.get("id") or f"index-{tidx}"
        task_name = task.get("name") or "(未命名任务)"
        auto_run = bool(task.get("auto_run", False))

        entry["actions"].append(
            f"准备推进：[{stage.get('name') or stage.get('id')}] {task_id} - {task_name} (auto_run={auto_run})"
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
            if task_is_done(task):
                entry["actions"].append(
                    f"人工检查点已落定（{task.get('status')}），无需自动执行：{task_id}"
                )
            else:
                entry["actions"].append(f"人工检查点，跳过自动完成：{task_id}")

        stage_done = stage_is_done(stage)
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

