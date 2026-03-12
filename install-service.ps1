# Engram Windows Service Setup — run as Administrator
# Creates a scheduled task that starts Engram on logon

$taskName = "Engram"
$nodeExe = (Get-Command node -ErrorAction SilentlyContinue).Source
if (-not $nodeExe) {
    Write-Host "ERROR: Node.js not found in PATH" -ForegroundColor Red
    exit 1
}

# Remove existing task if present
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

$action = New-ScheduledTaskAction `
    -Execute $nodeExe `
    -Argument "--experimental-strip-types server.ts" `
    -WorkingDirectory "C:\Users\Zan\engram"

$trigger = New-ScheduledTaskTrigger -AtLogon -User "Zan"

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -RestartCount 5 `
    -ExecutionTimeLimit (New-TimeSpan -Days 365) `
    -StartWhenAvailable

$principal = New-ScheduledTaskPrincipal -UserId "Zan" -LogonType Interactive -RunLevel Limited

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Engram v5.3 persistent memory server"

# Set environment variable for the task
$task = Get-ScheduledTask -TaskName $taskName
Write-Host ""
Write-Host "Scheduled task '$taskName' created." -ForegroundColor Green
Write-Host "Engram will start automatically at logon." -ForegroundColor Cyan
Write-Host ""
Write-Host "To start now:  schtasks /run /tn Engram" -ForegroundColor Yellow
Write-Host "To stop:       schtasks /end /tn Engram" -ForegroundColor Yellow
Write-Host "To check:      curl http://127.0.0.1:4200/health" -ForegroundColor Yellow
