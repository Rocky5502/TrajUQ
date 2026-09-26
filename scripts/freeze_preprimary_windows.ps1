param(
    [string]$Python = "python",
    [switch]$SkipCpuInstall
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Run-Step {
    param([string]$Name, [scriptblock]$Command)
    Write-Host ""
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
}

New-Item -ItemType Directory -Force -Path results | Out-Null

$Dirty = (& git status --porcelain) -join "`n"
if ($LASTEXITCODE -ne 0) { throw "Unable to inspect git worktree." }
if ($Dirty) {
    throw "Worktree is dirty. Commit or stash changes before freezing the experiment."
}

$Head = (& git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or -not $Head) {
    throw "Unable to resolve current git commit."
}
Write-Host "FROZEN_GIT_COMMIT=$Head" -ForegroundColor Yellow

if (-not $SkipCpuInstall) {
    Run-Step "Install CPU/dev verification dependencies" {
        & $Python -m pip install -e ".[dev,math]"
    }
}

Run-Step "Clean-commit CPU verification" {
    & $Python scripts/verify_cpu.py --output results/cpu_verification.json --logs-dir results/cpu_verification_logs
}

Run-Step "Resolve immutable primary model/dataset revisions" {
    & $Python scripts/resolve_experiment_revisions.py --output results/revision_ledger.json --frozen-model-dir results/frozen_models
}

$Cpu = Get-Content results/cpu_verification.json -Raw | ConvertFrom-Json
$Ledger = Get-Content results/revision_ledger.json -Raw | ConvertFrom-Json

if ($Cpu.status -ne "PASS") { throw "CPU verification did not PASS." }
if ($Ledger.status -ne "PASS") { throw "Revision ledger did not PASS." }
if ($Ledger.protocol -ne "trajuq_revision_ledger_v2") {
    throw "Unexpected revision-ledger protocol: $($Ledger.protocol)"
}
if ($Cpu.git_commit -ne $Head) {
    throw "CPU verification commit $($Cpu.git_commit) does not match frozen HEAD $Head."
}
if ($Ledger.git_commit -ne $Head) {
    throw "Revision-ledger commit $($Ledger.git_commit) does not match frozen HEAD $Head."
}

Run-Step "Refresh project status" {
    & $Python scripts/project_status.py
}

Write-Host ""
Write-Host "PRE-PRIMARY FREEZE PASS" -ForegroundColor Green
Write-Host "Frozen commit: $Head"
Write-Host "CPU artifact: results/cpu_verification.json"
Write-Host "Revision ledger: results/revision_ledger.json"
Write-Host "Frozen model configs: results/frozen_models/"
Write-Host ""
Write-Host "Next: do not edit tracked experiment code. Run LLaDA parity from this same commit:"
Write-Host "  PowerShell -ExecutionPolicy Bypass -File scripts/pilot_windows.ps1 -Stage prepare"
Write-Host "  PowerShell -ExecutionPolicy Bypass -File scripts/pilot_windows.ps1 -Stage parity"
