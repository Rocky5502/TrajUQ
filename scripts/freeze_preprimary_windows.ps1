param(
    [string]$Python = "python",
    [switch]$SkipDependencyInstall
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

if (-not $SkipDependencyInstall) {
    Run-Step "Install CPU/model/data verification dependencies" {
        & $Python -m pip install -e ".[dev,math,models]"
    }
}

Run-Step "Clean-commit CPU verification" {
    & $Python scripts/verify_cpu.py --output results/cpu_verification.json --logs-dir results/cpu_verification_logs
}

Run-Step "Resolve immutable primary model/dataset revisions" {
    & $Python scripts/resolve_experiment_revisions.py --output results/revision_ledger.json --frozen-model-dir results/frozen_models
}

Run-Step "Verify frozen gated-dataset access" {
    & $Python scripts/verify_gated_dataset_access.py --revision-ledger results/revision_ledger.json --output results/gated_dataset_access.json
}

$Cpu = Get-Content results/cpu_verification.json -Raw | ConvertFrom-Json
$Ledger = Get-Content results/revision_ledger.json -Raw | ConvertFrom-Json
$Gated = Get-Content results/gated_dataset_access.json -Raw | ConvertFrom-Json

if ($Cpu.protocol -ne "trajuq_cpu_verification_v1") {
    throw "Unexpected CPU-verification protocol: $($Cpu.protocol)"
}
if ($Cpu.status -ne "PASS") { throw "CPU verification did not PASS." }
if ($Cpu.git_clean -ne $true) { throw "CPU verification was not recorded from a clean worktree." }
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

if ($Gated.protocol -ne "trajuq_gated_dataset_access_v1") {
    throw "Unexpected gated-dataset-access protocol: $($Gated.protocol)"
}
if ($Gated.status -ne "PASS") { throw "Gated dataset access did not PASS." }
if ($Gated.git_clean -ne $true) {
    throw "Gated dataset access was not recorded from a clean worktree."
}
if ($Gated.git_commit -ne $Head) {
    throw "Gated dataset access commit $($Gated.git_commit) does not match frozen HEAD $Head."
}
$LedgerHash = (Get-FileHash results/revision_ledger.json -Algorithm SHA256).Hash.ToLowerInvariant()
if ($Gated.revision_ledger_sha256 -ne $LedgerHash) {
    throw "Gated dataset access is not bound to the current revision ledger."
}

$DirtyAfterFreeze = (& git status --porcelain) -join "`n"
if ($LASTEXITCODE -ne 0) { throw "Unable to re-check git worktree after freeze." }
if ($DirtyAfterFreeze) {
    throw "Tracked worktree changed during the pre-primary freeze. Do not trust the frozen artifacts."
}

Run-Step "Refresh project status" {
    & $Python scripts/project_status.py
}

Run-Step "Export private pre-primary handoff" {
    & $Python scripts/export_research_handoff.py --output dist/trajuq_preprimary_handoff.zip --skip-status-refresh
}

Write-Host ""
Write-Host "PRE-PRIMARY FREEZE PASS" -ForegroundColor Green
Write-Host "Frozen commit: $Head"
Write-Host "CPU artifact: results/cpu_verification.json"
Write-Host "Revision ledger: results/revision_ledger.json"
Write-Host "Gated dataset access: results/gated_dataset_access.json"
Write-Host "Frozen model configs: results/frozen_models/"
Write-Host "Private handoff: dist/trajuq_preprimary_handoff.zip"
Write-Host ""
Write-Host "Next: do not edit tracked experiment code. Run LLaDA parity from this same commit:"
Write-Host "  PowerShell -ExecutionPolicy Bypass -File scripts/pilot_windows.ps1 -Stage prepare"
Write-Host "  PowerShell -ExecutionPolicy Bypass -File scripts/pilot_windows.ps1 -Stage parity"
