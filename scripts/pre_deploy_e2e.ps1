param(
    [string]$PythonPath = "",
    [string]$ImageTag = "openenv-rl:predeploy",
    [int]$ContainerPort = 8786,
    [int]$StartupTimeoutSec = 120,
    [switch]$Quick,
    [switch]$SkipFrontendBuild,
    [switch]$SkipDockerBuild,
    [switch]$SkipDockerRuntime,
    [switch]$SkipOpenEnvCli
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:StepResults = New-Object System.Collections.Generic.List[object]

function Add-StepResult {
    param(
        [string]$Name,
        [string]$Status,
        [double]$DurationSec,
        [string]$Detail = ""
    )

    $script:StepResults.Add([pscustomobject]@{
        Step        = $Name
        Status      = $Status
        DurationSec = [Math]::Round($DurationSec, 2)
        Detail      = $Detail
    }) | Out-Null
}

function Show-Summary {
    Write-Host ""
    Write-Host "=============================================="
    Write-Host "Pre-Deploy E2E Summary"
    Write-Host "=============================================="

    $table = $script:StepResults | Select-Object Step, Status, DurationSec, Detail
    if ($table.Count -gt 0) {
        $table | Format-Table -AutoSize | Out-String | Write-Host
    }

    $failed = @($script:StepResults | Where-Object { $_.Status -eq "FAILED" })
    if ($failed.Count -gt 0) {
        Write-Host "Result: FAILED ($($failed.Count) step(s) failed)" -ForegroundColor Red
    }
    else {
        Write-Host "Result: PASSED (all checks succeeded)" -ForegroundColor Green
    }
}

function Ensure-CommandExists {
    param([string[]]$Candidates)

    foreach ($candidate in $Candidates) {
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($null -ne $cmd) {
            return $cmd.Source
        }
    }

    throw "Required command not found. Tried: $($Candidates -join ', ')"
}

function Resolve-PythonExe {
    param([string]$RequestedPath)

    if ($RequestedPath) {
        if (Test-Path $RequestedPath) {
            return (Resolve-Path $RequestedPath).Path
        }
        throw "PythonPath was provided but not found: $RequestedPath"
    }

    $candidatePaths = @(
        ".venv313\\Scripts\\python.exe",
        ".venv\\Scripts\\python.exe"
    )

    foreach ($candidate in $candidatePaths) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    $pythonCmd = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($null -ne $pythonCmd) {
        return $pythonCmd.Source
    }

    throw "Could not resolve Python interpreter. Provide -PythonPath explicitly."
}

function Invoke-CheckedCommand {
    param(
        [string]$Executable,
        [string[]]$Arguments
    )

    Write-Host "-> $Executable $($Arguments -join ' ')"
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE: $Executable $($Arguments -join ' ')"
    }
}

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Action
    )

    Write-Host ""
    Write-Host "=== $Name ===" -ForegroundColor Cyan

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        & $Action
        $sw.Stop()
        Add-StepResult -Name $Name -Status "PASSED" -DurationSec $sw.Elapsed.TotalSeconds
        Write-Host "[PASS] $Name" -ForegroundColor Green
    }
    catch {
        $sw.Stop()
        Add-StepResult -Name $Name -Status "FAILED" -DurationSec $sw.Elapsed.TotalSeconds -Detail $_.Exception.Message
        Write-Host "[FAIL] $Name" -ForegroundColor Red
        Write-Host "Reason: $($_.Exception.Message)" -ForegroundColor Red
        Show-Summary
        throw
    }
}

function Wait-ForHealth {
    param(
        [string]$HealthUrl,
        [int]$TimeoutSec
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $lastError = "No response yet"

    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-RestMethod -Method Get -Uri $HealthUrl -TimeoutSec 5
            return $response
        }
        catch {
            $lastError = $_.Exception.Message
            Start-Sleep -Seconds 2
        }
    }

    throw "Timed out waiting for container health endpoint at $HealthUrl. Last error: $lastError"
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Host "Repo root: $repoRoot"

$resolvedPython = $null
$npmExecutable = $null
$dockerExecutable = $null

Invoke-Step -Name "Resolve toolchain" -Action {
    $resolvedPython = Resolve-PythonExe -RequestedPath $PythonPath
    Write-Host "Python: $resolvedPython"

    if (-not $SkipFrontendBuild) {
        $npmExecutable = Ensure-CommandExists -Candidates @("npm.cmd", "npm")
        Write-Host "NPM: $npmExecutable"
    }

    if (-not $SkipDockerBuild -or -not $SkipDockerRuntime) {
        $dockerExecutable = Ensure-CommandExists -Candidates @("docker")
        Write-Host "Docker: $dockerExecutable"
    }
}

Invoke-Step -Name "Python syntax and import sanity" -Action {
    Invoke-CheckedCommand -Executable $resolvedPython -Arguments @("-m", "compileall", "app", "rl", "scripts", "tests")
    Invoke-CheckedCommand -Executable $resolvedPython -Arguments @("-c", "import fastapi, uvicorn; print('python runtime ok')")
}

Invoke-Step -Name "OpenEnv manifest and import validation" -Action {
    $args = @("scripts/validate_env.py", "--repo", ".")
    if ($SkipOpenEnvCli) {
        $args += "--skip-openenv-cli"
    }
    Invoke-CheckedCommand -Executable $resolvedPython -Arguments $args
}

Invoke-Step -Name "Deterministic smoke baseline" -Action {
    Invoke-CheckedCommand -Executable $resolvedPython -Arguments @("scripts/smoke_test.py")
}

Invoke-Step -Name "API contract E2E suite" -Action {
    Invoke-CheckedCommand -Executable $resolvedPython -Arguments @("-m", "pytest", "tests/test_api_end_to_end_suite.py", "-v", "--tb=short")
}

if (-not $Quick) {
    Invoke-Step -Name "Core API and environment regression tests" -Action {
        Invoke-CheckedCommand -Executable $resolvedPython -Arguments @(
            "-m", "pytest",
            "tests/test_phase1_models.py",
            "tests/test_phase1_sector_and_tasks.py",
            "tests/test_phase1_event_engine.py",
            "tests/test_phase1_signal_computer.py",
            "tests/test_phase2_env_integration.py",
            "tests/test_phase2_simulator.py",
            "tests/test_phase2_api.py",
            "tests/test_live_simulation_e2e.py",
            "tests/test_action_mask.py",
            "-v",
            "--tb=short"
        )
    }
}

if (-not $SkipFrontendBuild) {
    Invoke-Step -Name "Frontend install and production build" -Action {
        Invoke-CheckedCommand -Executable $npmExecutable -Arguments @("--prefix", "frontend/react", "ci", "--no-audit", "--no-fund")
        Invoke-CheckedCommand -Executable $npmExecutable -Arguments @("--prefix", "frontend/react", "run", "build")
    }
}

if (-not $SkipDockerBuild) {
    Invoke-Step -Name "Docker image build" -Action {
        Invoke-CheckedCommand -Executable $dockerExecutable -Arguments @("build", "-t", $ImageTag, ".")
    }
}

if (-not $SkipDockerRuntime) {
    Invoke-Step -Name "Docker runtime endpoint sanity" -Action {
        $containerName = "openenv-preflight-" + [Guid]::NewGuid().ToString("N").Substring(0, 8)
        $healthUrl = "http://127.0.0.1:$ContainerPort/health"
        $baseUrl = "http://127.0.0.1:$ContainerPort"
        $containerStarted = $false

        try {
            $runOutput = & $dockerExecutable run -d --rm --name $containerName -p "$ContainerPort`:7860" $ImageTag
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to start Docker container $containerName"
            }
            $containerStarted = $true
            Write-Host "Container: $containerName"
            Write-Host "Container ID: $($runOutput | Select-Object -Last 1)"

            $health = Wait-ForHealth -HealthUrl $healthUrl -TimeoutSec $StartupTimeoutSec
            if ($health.status -notin @("ok", "degraded")) {
                throw "Unexpected health status: $($health.status)"
            }

            $resetBody = @{ task_id = "district_backlog_easy"; seed = 42 } | ConvertTo-Json
            $reset = Invoke-RestMethod -Method Post -Uri "$baseUrl/reset" -ContentType "application/json" -Body $resetBody -TimeoutSec 20
            if (-not $reset.session_id) {
                throw "Reset response missing session_id"
            }

            $stepBody = @{
                session_id = $reset.session_id
                action = @{ action_type = "advance_time" }
            } | ConvertTo-Json -Depth 5
            $step = Invoke-RestMethod -Method Post -Uri "$baseUrl/step" -ContentType "application/json" -Body $stepBody -TimeoutSec 20
            if (-not $step.observation) {
                throw "Step response missing observation"
            }

            $gradeBody = @{ session_id = $reset.session_id } | ConvertTo-Json
            $grade = Invoke-RestMethod -Method Post -Uri "$baseUrl/grade" -ContentType "application/json" -Body $gradeBody -TimeoutSec 20
            $score = [double]$grade.score
            if ($score -lt 0.0 -or $score -gt 1.0) {
                throw "Grade score out of range: $score"
            }

            Write-Host "Health status: $($health.status)"
            Write-Host "Session ID: $($reset.session_id)"
            Write-Host "Grade score: $score"
        }
        finally {
            if ($containerStarted) {
                try {
                    & $dockerExecutable stop $containerName | Out-Null
                }
                catch {
                    Write-Warning "Failed to stop container $containerName: $($_.Exception.Message)"
                }
            }
        }
    }
}

Show-Summary
Write-Host "Pre-deployment E2E checks completed successfully." -ForegroundColor Green
exit 0
