# PowerShell script for training with logging
Write-Host "Starting IMM Debug Training with Logging..." -ForegroundColor Green
Write-Host ""

# Create log directory if it doesn't exist
$logDir = "data\logs\debug_perceptual"
if (!(Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force
    Write-Host "Created log directory: $logDir" -ForegroundColor Yellow
}

# Generate timestamp for log file
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "$logDir\training_log_$timestamp.txt"

Write-Host "Log file: $logFile" -ForegroundColor Cyan
Write-Host "Start time: $(Get-Date)" -ForegroundColor Cyan
Write-Host ""

# Start training process
$process = Start-Process -FilePath "python" -ArgumentList "debug_train_perceptual.py" -RedirectStandardOutput $logFile -RedirectStandardError $logFile -PassThru -NoNewWindow

Write-Host "Training started with PID: $($process.Id)" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop training and view log" -ForegroundColor Yellow
Write-Host ""

# Monitor the process
try {
    $process.WaitForExit()
    Write-Host ""
    Write-Host "Training completed!" -ForegroundColor Green
    Write-Host "Return code: $($process.ExitCode)" -ForegroundColor Cyan
} catch {
    Write-Host ""
    Write-Host "Training interrupted!" -ForegroundColor Red
    $process.Kill()
}

Write-Host ""
Write-Host "Log saved to: $logFile" -ForegroundColor Cyan
Write-Host "Press any key to view the log file..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Open log file in default text editor
Start-Process $logFile


