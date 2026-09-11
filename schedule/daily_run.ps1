# daily_run.ps1
# 用法示例（命令行）：
#   powershell -ExecutionPolicy Bypass -File "E:\各种PY程序\16_金融市场预测模型\schedule\daily_run.ps1"

$ErrorActionPreference = "Continue"

$root = Split-Path $MyInvocation.MyCommand.Path -Parent
$proj = Split-Path $root -Parent
$planPath = Join-Path $root "plan.json"
$logDir  = Join-Path $root "logs"
$logFile = Join-Path $logDir ("run_" + (Get-Date -Format "yyyy-MM-dd") + ".json")

if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
if (-not (Test-Path $planPath)) {
    Write-Host "[daily] plan.json 不存在，无法推进"
    exit 1
}

$plan = Get-Content $planPath -Raw | ConvertFrom-Json

$logEntry = @{
  run_date = (Get-Date -Format "yyyy-MM-dd")
  ran_at   = (Get-Date -Format "o")
  stage    = $plan.current_stage_index
  task     = $plan.current_task_index
  actions  = @()
  result   = "unknown"
  blockers = @()
}

function Write-LogEntry {
    param($logEntry)
    # 上一版漏定义该函数：PowerShell 路径下走到这里必抛 CommandNotFoundException，
    # 结果是 plan.json / run_<date>.json 都不落盘（与 run_daily.py 的日志契约一致）。
    try {
        if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
        $logEntry | ConvertTo-Json -Depth 6 | Set-Content -Path $logFile -Encoding UTF8
    }
    catch {
        Write-Host "[daily] 日志写入失败：$_"
    }
}

function Try-RunVectorbtSmokeTest {
    param($proj, $logEntry)
    $vbtDir = Join-Path $proj "libs" "vectorbt"
    if (-not (Test-Path $vbtDir)) {
        try {
            git clone --depth 1 https://github.com/vectorbt/vectorbt.git $vbtDir 2>&1 | Out-Null
        } catch {
            return $false
        }
    }
    $hasReadme = (Test-Path (Join-Path $vbtDir "README.md"))
    $hasExamples = (Test-Path (Join-Path $vbtDir "examples"))
    return ($hasReadme -or $hasExamples)
}

function Try-RunTsaiSmokeTest {
    param($proj, $logEntry)
    $tsaiDir = Join-Path $proj "libs" "tsai"
    if (-not (Test-Path $tsaiDir)) {
        try {
            git clone --depth 1 https://github.com/timeseriesAI/tsai.git $tsaiDir 2>&1 | Out-Null
        } catch {
            return $false
        }
    }
    $hasReadme = (Test-Path (Join-Path $tsaiDir "README.md"))
    $hasNotebooks = (Test-Path (Join-Path $tsaiDir "notebooks"))
    return ($hasReadme -or $hasNotebooks)
}

function Try-RunFinRLSmokeTest {
    param($proj, $logEntry)
    # 与 Try-RunVectorbtSmokeTest / Try-RunTsaiSmokeTest 同构；
    # 上一版 T5.1 分支调用了未定义的本函数，PowerShell 路径同样会中断。
    $dir = Join-Path $proj "libs" "FinRL"
    if (-not (Test-Path $dir)) {
        try {
            git clone --depth 1 https://github.com/AI4Finance-Foundation/FinRL.git $dir 2>&1 | Out-Null
        } catch {
            return $false
        }
    }
    $hasReadme = (Test-Path (Join-Path $dir "README.md"))
    $hasExamples = (Test-Path (Join-Path $dir "examples"))
    return ($hasReadme -or $hasExamples)
}

try {
    $stage = $plan.stages[$plan.current_stage_index]

    if ($stage.status -eq "completed") {
        $found = $false
        for ($i = 0; $i -lt $plan.stages.Count; $i++) {
            if ($plan.stages[$i].status -eq "pending") {
                $plan.current_stage_index = $i
                $plan.current_task_index  = 0
                $found = $true
                break
            }
        }
        if (-not $found) {
            $logEntry.result = "done_all_stages"
            $logEntry.actions += "所有阶段已完成"
            Write-LogEntry $logEntry
            $plan | ConvertTo-Json -Depth 6 | Set-Content $planPath
            exit 0
        }
        $stage = $plan.stages[$plan.current_stage_index]
    }

    $task = $stage.tasks[$plan.current_task_index]
    $stageName = $stage.name
    $taskId    = $task.id
    $taskName  = $task.name
    $autoRun   = [bool]$task.auto_run

    $logEntry.actions += "准备推进：[$stageName] $taskId - $taskName (auto_run=$autoRun)"

    if ($stage.status -eq "pending") {
        $stage.status = "in_progress"
        $stage.started_at = (Get-Date -Format "o")
        $logEntry.actions += "阶段开始：$stageName"
    }

    if ($autoRun) {
        $ok = $false
        $msg = ""

        if ($taskId -eq "T1.1") {
            $ok = Try-RunVectorbtSmokeTest $proj $logEntry
            $msg = if ($ok) { "vectorbt 最小示例尝试完成" } else { "vectorbt 最小示例尝试失败/未达标" }
        }
        elseif ($taskId -eq "T1.3") {
            $ok = Test-Path (Join-Path $proj "backtest" "run_minimal.py")
            $msg = if ($ok) { "最小回测脚本已存在" } else { "最小回测脚本尚未创建" }
        }
        elseif ($taskId -eq "T2.3") {
            $ok = Try-RunTsaiSmokeTest $proj $logEntry
            $msg = if ($ok) { "tsai 示例尝试完成" } else { "tsai 示例尝试失败/未达标" }
        }
        elseif ($taskId -eq "T2.4") {
            $ok = (Test-Path (Join-Path $proj "models" "tsai_baselines"))
            $msg = if ($ok) { "tsai baseline 目录已存在" } else { "tsai baseline 目录尚未创建" }
        }
        elseif ($taskId -eq "T3.3") {
            $ok = (Test-Path (Join-Path $proj "pipeline" "mlfinlab" "experiments"))
            $msg = if ($ok) { "mlfinlab 试跑目录已存在" } else { "mlfinlab 试跑目录尚未创建" }
        }
        elseif ($taskId -eq "T5.1") {
            $ok = Try-RunFinRLSmokeTest $proj $logEntry
            $msg = if ($ok) { "FinRL 示例尝试完成" } else { "FinRL 示例尝试失败/未达标" }
        }
        elseif ($taskId -eq "T5.4") {
            $ok = (Test-Path (Join-Path $proj "rl" "finrl_experiments" "results_and_decision.md"))
            $msg = if ($ok) { "FinRL 结论文件已存在" } else { "FinRL 结论文件尚未创建" }
        }
        else {
            $msg = "未知可自动任务 $taskId"
        }

        $logEntry.actions += "自动任务结果：$msg"

        if ($ok) {
            $task.status = "completed"
            $task.completed_at = (Get-Date -Format "o")
            $logEntry.actions += "标记完成：$taskId"
        }
        else {
            $logEntry.blockers += "$taskId - $msg"
            $plan.blockers += "$taskId - $msg"
        }
    }
    else {
        $logEntry.actions += "人工检查点，跳过自动完成：$taskId"
    }

    $stageDone = $true
    foreach ($t in $stage.tasks) {
        if ($t.status -ne "completed") {
            $stageDone = $false
            break
        }
    }
    if ($stageDone) {
        $stage.status = "completed"
        $stage.completed_at = (Get-Date -Format "o")
        $logEntry.actions += "阶段完成：$stageName"

        if ($plan.current_stage_index -lt $plan.stages.Count - 1) {
            $plan.current_stage_index++
            $plan.current_task_index = 0
        }
        else {
            $logEntry.result = "all_stages_completed"
        }
    }

    $plan.last_run_date = (Get-Date -Format "yyyy-MM-dd")
    $plan.next_run_date = (Get-Date).AddDays(1).ToString("yyyy-MM-dd")
    if ($logEntry.blockers.Count -gt 0) {
        $plan.blockers = @($plan.blockers | Select-Object -Unique)
    }

    $plan | ConvertTo-Json -Depth 6 | Set-Content $planPath
    $logEntry.result = "advanced"
    Write-LogEntry $logEntry
}
catch {
    $logEntry.result = "error"
    $logEntry.actions += "异常：$_"
    $logEntry.blockers += "runtime_error: $_"
    $plan.blockers += "runtime_error: $_"
    $plan | ConvertTo-Json -Depth 6 | Set-Content $planPath
    Write-LogEntry $logEntry
    exit 1
}
