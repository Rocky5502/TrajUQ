param(
    [string]$Python = "python",
    [switch]$SkipInstall,
    [switch]$AllowDirty
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# scripts/verify_cpu.py is the authoritative CPU verifier and runs compileall,
# Ruff, pytest, smoke_metrics.py, smoke_end_to_end.py, audit_math500_scorer.py,
# and audit_paper.py.

function Run-Step {
    param([string]$Name, [scriptblock]$Command)
    Write-Host ""
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
}

if (-not $SkipInstall) {
    Run-Step "Install CPU/dev dependencies" {
        & $Python -m pip install -e ".[dev,math]"
    }
}

$VerifyArgs = @(
    "scripts/verify_cpu.py",
    "--output", "results/cpu_verification.json",
    "--logs-dir", "results/cpu_verification_logs"
)
if ($AllowDirty) {
    $VerifyArgs += "--allow-dirty"
}

Run-Step "TrajUQ CPU verification" {
    & $Python @VerifyArgs
}

Write-Host ""
Write-Host "LOCAL CPU VERIFICATION PASS" -ForegroundColor Green
Write-Host "Artifact: results/cpu_verification.json"
Write-Host "This does not replace model-specific GPU parity gates."
