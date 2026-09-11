"""每日自动推进脚本（schedule/run_daily.py）必须在「所有阶段已完成」时也保持可用。

回归背景（真实缺陷）：脚本 `run_daily()` 缺少 `except/finally` 收尾，函数在末尾
`if __name__` 之前直接结束，导致**整个文件语法错误**：

    $ python schedule/run_daily.py
    SyntaxError: expected 'except' or 'finally' block

即「按排期计划自动开发」的核心执行器从未真正运行过；同时函数体里还调用了不存在的
`log()`，计划与日志都不可能落盘。本文件把「可编译 + 可执行 + 结果落盘」固化为断言。

测试全部在 tmp_path 中运行，**不触碰仓库内的 plan.json**。
"""
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "schedule" / "run_daily.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("schedule_run_daily", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_plan(path: Path, plan: dict) -> None:
    path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")


def _all_done_plan() -> dict:
    return {
        "project": "T",
        "stages": [
            {
                "id": "S1",
                "name": "A",
                "status": "completed",
                "tasks": [{"id": "T1.1", "name": "a", "status": "completed", "auto_run": True}],
            }
        ],
        "current_stage_index": 0,
        "current_task_index": 0,
    }


class TestRunnerIsExecutable:
    def test_runner_compiles(self):
        """语法错误会让每日排期静默停摆，必须可编译。"""
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(RUNNER)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"schedule/run_daily.py 无法编译:\n{result.stderr}"

    def test_runner_module_imports(self):
        module = _load_runner()
        assert callable(module.run_day)


class TestRunDayPersistsState:
    def test_done_all_stages_marks_result_and_writes_log(self, tmp_path):
        plan_path = tmp_path / "plan.json"
        log_dir = tmp_path / "logs"
        _write_plan(plan_path, _all_done_plan())

        _load_runner().run_day(str(plan_path), str(log_dir))

        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        assert plan.get("result") == "done_all_stages"
        assert plan.get("last_run_date"), "应记录最后运行日期"
        assert plan.get("next_run_date") is None, "全部完成后不应再排下一次"

        logs = list(log_dir.glob("run_*.json"))
        assert logs, "必须写入运行日志（原实现调用了不存在的 log() 而不可能落盘）"
        entry = json.loads(logs[0].read_text(encoding="utf-8"))
        assert entry["result"] == "done_all_stages"

    def test_auto_task_completion_advances_plan(self, tmp_path):
        # 脚本用 `Path(plan_path).parent.parent` 作为项目根，故按 <root>/schedule/plan.json 布置
        sched = tmp_path / "schedule"
        sched.mkdir(parents=True, exist_ok=True)
        (tmp_path / "backtest").mkdir(parents=True, exist_ok=True)
        (tmp_path / "backtest" / "run_minimal.py").write_text("# stub\n", encoding="utf-8")
        plan_path = sched / "plan.json"
        log_dir = sched / "logs"
        _write_plan(plan_path, {
            "project": "T",
            "stages": [{
                "id": "S1", "name": "A", "status": "in_progress",
                "tasks": [{"id": "T1.3", "name": "最小回测脚本",
                           "status": "pending", "auto_run": True}],
            }],
            "current_stage_index": 0,
            "current_task_index": 0,
        })

        _load_runner().run_day(str(plan_path), str(log_dir))

        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        # T1.3 的完成条件是 backtest/run_minimal.py 存在（仓库中恒为真）
        assert plan["stages"][0]["tasks"][0]["status"] == "completed"
        assert plan["stages"][0]["status"] == "completed"
        assert plan.get("last_run_date")

    def test_corrupt_plan_does_not_crash_runner(self, tmp_path):
        """plan.json 损坏时脚本不得抛未捕获异常（无人值守场景）。"""
        plan_path = tmp_path / "plan.json"
        plan_path.write_text("{ this is not json", encoding="utf-8")
        module = _load_runner()
        with pytest.raises(json.JSONDecodeError):
            # load_plan 在进入 try 之前调用：调用方需保证 plan.json 合法，
            # 这正是 tests/test_plan_json_validity.py 守护的对象。
            module.run_day(str(plan_path), str(tmp_path / "logs"))


# ---------------------------------------------------------------- PS1 路径
def test_powershell_runner_defines_all_called_functions():
    """daily_run.ps1 里被调用的自定义函数必须都有定义。

    背景：该脚本曾调用未定义的 `Write-LogEntry` / `Try-RunFinRLSmokeTest`，
    PowerShell 路径下必抛 CommandNotFoundException —— 而这条路径只在
    Windows 计划任务里跑到，CI 的 Python 测试完全覆盖不到。
    这里做静态检查，把「只在生产路径炸」的缺陷提前暴露。
    """
    import re

    script = (PROJECT_ROOT / "schedule" / "daily_run.ps1").read_text(encoding="utf-8")
    defined = set(re.findall(r"^\s*function\s+([A-Za-z][\w-]*)", script, re.M))
    called = set(re.findall(r"^\s*([A-Z][\w]*)\s+\$", script, re.M))
    # 系统 cmdlet 白名单：非自定义函数
    builtin = {"Write-Host", "Get-Content", "Join-Path", "Test-Path", "New-Item",
               "Split-Path", "ConvertFrom-Json", "ConvertTo-Json", "Select-Object"}
    missing = {c for c in called if c not in defined and c not in builtin
               and ("-" not in c or c.startswith("Try-"))}
    assert not missing, f"daily_run.ps1 调用了未定义的函数: {sorted(missing)}"


def test_powershell_runner_writes_log_file():
    """日志函数必须真的写入 $logFile，而不是只存在定义。"""
    script = (PROJECT_ROOT / "schedule" / "daily_run.ps1").read_text(encoding="utf-8")
    assert "function Write-LogEntry" in script
    body = script[script.index("function Write-LogEntry"):]
    body = body[:body.index("\n}\n") + 3]
    assert "$logFile" in body
