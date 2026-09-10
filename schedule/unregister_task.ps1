# unregister_task.ps1
# 注销每日自动推进计划任务
# 用法：以管理员身份运行 PowerShell，然后执行：
#   powershell -ExecutionPolicy Bypass -File "E:\各种PY程序\16_金融市场预测模型\schedule\unregister_task.ps1"

$taskName = "FinancialModeling_DailyAdvance"

try {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "[OK] 计划任务已注销：$taskName"
}
catch {
    Write-Host "[ERROR] 注销失败：$_"
    exit 1
}
