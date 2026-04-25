param(
    [string]$PythonPath = "C:\Users\siddh\OPENENV_RL\.venv313\Scripts\python.exe",
    [switch]$Full
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not (Test-Path $PythonPath)) {
    throw "Python not found at: $PythonPath"
}

Write-Host "Repo root: $repoRoot"
Write-Host "Python:    $PythonPath"

Write-Host ""
Write-Host "Step 1/3: Syntax check"
& $PythonPath -m py_compile app\main.py tests\test_api_end_to_end_suite.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Step 2/3: Run E2E API suite"
& $PythonPath -m pytest tests\test_api_end_to_end_suite.py -v --tb=short
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Full) {
    Write-Host ""
    Write-Host "Step 3/3: Run full API regression suite"
    & $PythonPath -m pytest tests\test_api.py tests\test_api_end_to_end_suite.py -v --tb=short
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host ""
Write-Host "API E2E test run completed."
