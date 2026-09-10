# register_task.ps1
# 注册每日自动推进计划任务
# 用法：以管理员身份运行 PowerShell，然后执行：
#   powershell -ExecutionPolicy Bypass -File "E:\各种PY程序\16_金融市场预测模型\schedule\register_task.ps1"

$taskName = "FinancialModeling_DailyAdvance"
$scheduleDir = "E:\各种PY程序\16_金融市场预测模型\schedule"
$scriptPath = Join-Path $scheduleDir "daily_run.ps1"

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"$scriptPath`"" `
    -WorkingDirectory $scheduleDir

$trigger = New-ScheduledTaskTrigger -Daily -At "09:00"

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -RunLevel Highest

try {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    Register-ScheduledTask `
        -TaskName $taskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Description "金融市场预测模型 - 每日自动推进排期计划"

    Write-Host "[OK] 计划任务已注册：$taskName"
    Write-Host "  触发时间：每天 09:00"
    Write-Host "  执行脚本：$scriptPath"
    Write-Host "  工作目录：$scheduleDir"
}
catch {
    Write-Host "[ERROR] 注册失败：$_"
    exit 1
}
