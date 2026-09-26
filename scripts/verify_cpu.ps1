param(
    [string]$Python = "python",
    [switch]$SkipInstall
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

if (-not $SkipInstall) {
    Run-Step "Install CPU/dev dependencies" { & $Python -m pip install -e ".[dev,math]" }
}

Run-Step "Compile Python sources" { & $Python -m compileall -q src scripts tests }
Run-Step "Ruff" { & $Python -m ruff check src tests scripts }
Run-Step "Pytest" { & $Python -m pytest -q }
Run-Step "Metric smoke" { & $Python scripts/smoke_metrics.py }
Run-Step "End-to-end leak-safe smoke" { & $Python scripts/smoke_end_to_end.py --output-dir results/smoke_e2e_local }
Run-Step "Paper anonymity audit" { & $Python scripts/audit_paper.py }

Write-Host ""
Write-Host "LOCAL CPU VERIFICATION PASS" -ForegroundColor Green
Write-Host "This does not replace model-specific GPU parity gates."
