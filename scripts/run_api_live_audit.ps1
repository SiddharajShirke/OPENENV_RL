param(
    [string]$PythonPath = "C:\Users\siddh\OPENENV_RL\.venv313\Scripts\python.exe",
    [string]$BaseUrl = "http://127.0.0.1:7860",
    [int]$TimeoutSec = 30
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not (Test-Path $PythonPath)) {
    throw "Python not found at: $PythonPath"
}

Write-Host "Repo root: $repoRoot"
Write-Host "Python:    $PythonPath"
Write-Host "Base URL:  $BaseUrl"

Write-Host ""
Write-Host "Step 1/2: Syntax check"
& $PythonPath -m py_compile scripts\api_live_http_audit.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Step 2/2: Live HTTP endpoint audit"
& $PythonPath scripts\api_live_http_audit.py --base-url $BaseUrl --timeout-sec $TimeoutSec
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Live HTTP audit completed."
