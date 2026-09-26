param(
    [ValidateSet("prepare","parity","pilot","all")]
    [string]$Stage = "all",
    [string]$Python = "python"
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

function Remove-Artifact {
    param([string]$Path)
    Remove-Item $Path -ErrorAction SilentlyContinue
    Remove-Item "$Path.audit.json" -ErrorAction SilentlyContinue
}

function Read-FrozenLedger {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        throw "Missing $Path. First run scripts/freeze_preprimary_windows.ps1."
    }
    $Ledger = Get-Content $Path -Raw | ConvertFrom-Json
    if ($Ledger.status -ne "PASS") {
        throw "Revision ledger is not PASS."
    }
    if ($Ledger.protocol -ne "trajuq_revision_ledger_v2") {
        throw "Unexpected revision-ledger protocol: $($Ledger.protocol)"
    }
    $Head = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $Head) {
        throw "Unable to resolve current git commit."
    }
    if ($Ledger.git_commit -ne $Head) {
        throw "Revision ledger was frozen at $($Ledger.git_commit), but current HEAD is $Head. Do not mix commits."
    }
    return $Ledger
}

$ModelId = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"
$LedgerPath = "results/revision_ledger.json"
$ModelConfig = "results/frozen_models/llada.yaml"
$CalData = "data/processed/gsm8k/calibration.jsonl"
$TestData = "data/processed/gsm8k/test.jsonl"
$ParityReference = "runs/llada_parity_reference.jsonl"
$ParityTrajUQ = "runs/llada_parity_trajuq.raw.jsonl"
$ParityResult = "results/llada_parity.json"
$PilotRaw = "runs/llada_gsm8k_pilot50.raw.jsonl"
$PilotScored = "runs/llada_gsm8k_pilot50.scored.jsonl"

New-Item -ItemType Directory -Force -Path results | Out-Null
New-Item -ItemType Directory -Force -Path runs | Out-Null

$Ledger = Read-FrozenLedger $LedgerPath
$ModelEntry = @($Ledger.models | Where-Object { $_.adapter -eq "llada" })
$DatasetEntry = @($Ledger.datasets | Where-Object { $_.name -eq "gsm8k" })
if ($ModelEntry.Count -ne 1) { throw "Revision ledger must contain exactly one LLaDA entry." }
if ($DatasetEntry.Count -ne 1) { throw "Revision ledger must contain exactly one GSM8K entry." }

$ModelSha = [string]$ModelEntry[0].resolved_sha
$DatasetSha = [string]$DatasetEntry[0].resolved_sha
if (-not $ModelSha -or $ModelSha -eq "null") { throw "Ledger has no frozen LLaDA SHA." }
if (-not $DatasetSha -or $DatasetSha -eq "null") { throw "Ledger has no frozen GSM8K SHA." }
if (-not (Test-Path $ModelConfig)) { throw "Missing frozen LLaDA model config: $ModelConfig" }

Write-Host "FROZEN_GIT_COMMIT=$($Ledger.git_commit)" -ForegroundColor Yellow
Write-Host "MODEL_SHA=$ModelSha"
Write-Host "DATASET_SHA=$DatasetSha"

if ($Stage -in @("prepare", "all")) {
    Run-Step "Prepare frozen GSM8K pair" { & $Python scripts/prepare_eval_pair.py --dataset gsm8k --revision $DatasetSha }
    Run-Step "CUDA preflight" { & $Python scripts/preflight.py --model-config $ModelConfig --data $CalData --require-cuda }
    Run-Step "Environment snapshot" { & $Python scripts/environment_snapshot.py --output results/pilot_environment.json }
    Run-Step "Freeze manifest" { & $Python scripts/freeze_manifest.py --model-config $ModelConfig --dataset $CalData --dataset $TestData --output results/pilot_freeze_manifest.json }
}

if ($Stage -in @("parity", "pilot", "all")) {
    if (-not (Test-Path $CalData)) { throw "Missing $CalData. Run -Stage prepare first." }
}
if ($Stage -in @("pilot", "all")) {
    if (-not (Test-Path $TestData)) { throw "Missing $TestData. Run -Stage prepare first." }
}

if ($Stage -in @("parity", "all")) {
    Remove-Artifact $ParityReference
    Remove-Artifact $ParityTrajUQ
    Remove-Item $ParityResult -ErrorAction SilentlyContinue
    Remove-Item results/llada_instrumentation_overhead.json -ErrorAction SilentlyContinue

    Run-Step "20-prompt native reference" { & $Python scripts/run_llada_reference.py --model $ModelId --revision $ModelSha --input $CalData --output $ParityReference --limit 20 --steps 128 --max-new-tokens 128 --block-length 128 --temperature 0 }
    Run-Step "20-prompt TrajUQ telemetry" { & $Python scripts/run_generation.py --adapter llada --model $ModelId --revision $ModelSha --input $CalData --output $ParityTrajUQ --limit 20 --steps 128 --max-new-tokens 128 --block-length 128 --temperature 0 }
    Run-Step "Raw parity-run audit" { & $Python scripts/audit_run.py --run $ParityTrajUQ --stage raw --require-revision }
    Run-Step "Exact token parity" { & $Python scripts/compare_run_parity.py --reference $ParityReference --instrumented $ParityTrajUQ --require-token-ids --require-provenance --output $ParityResult }

    $Parity = Get-Content $ParityResult -Raw | ConvertFrom-Json
    if ($Parity.protocol -ne "trajuq_exact_parity_v3") { throw "Parity protocol is not v3. STOP." }
    if ($Parity.status -ne "PASS") { throw "Parity is not PASS. STOP: do not run the 50-item pilot." }
    if ($Parity.git_commit -ne $Ledger.git_commit) { throw "Parity commit differs from revision ledger. STOP." }
    if ($Parity.shared_records -ne 20) { throw "Parity did not cover exactly 20 shared prompts. STOP." }
    if ($Parity.run_provenance.model_revision -ne $ModelSha) { throw "Parity model revision differs from revision ledger. STOP." }

    Run-Step "Instrumentation overhead" { & $Python scripts/compare_instrumentation_overhead.py --reference $ParityReference --instrumented $ParityTrajUQ --output results/llada_instrumentation_overhead.json }
}

if ($Stage -in @("pilot", "all")) {
    if (-not (Test-Path $ParityResult)) { throw "Missing $ParityResult. Run -Stage parity first." }
    $Parity = Get-Content $ParityResult -Raw | ConvertFrom-Json
    if ($Parity.protocol -ne "trajuq_exact_parity_v3") { throw "Parity protocol is not v3. STOP." }
    if ($Parity.status -ne "PASS") { throw "Parity is not PASS. STOP." }
    if ($Parity.git_commit -ne $Ledger.git_commit) { throw "Parity commit differs from revision ledger. STOP." }
    if ($Parity.run_provenance.model_revision -ne $ModelSha) { throw "Parity model revision differs from revision ledger. STOP." }

    Remove-Artifact $PilotRaw
    Remove-Artifact $PilotScored
    Remove-Item results/llada_gsm8k_pilot50_gate.json -ErrorAction SilentlyContinue
    Remove-Item results/llada_gsm8k_pilot50_trace_review.md -ErrorAction SilentlyContinue

    Run-Step "50-item TrajUQ pilot" { & $Python scripts/run_generation.py --adapter llada --model $ModelId --revision $ModelSha --input $CalData --output $PilotRaw --limit 50 --steps 128 --max-new-tokens 128 --block-length 128 --temperature 0 }
    Run-Step "Raw pilot audit" { & $Python scripts/audit_run.py --run $PilotRaw --stage raw --require-revision }
    Run-Step "Score pilot into separate artifact" { & $Python scripts/score_outputs.py --run $PilotRaw --input $CalData --output $PilotScored }
    Run-Step "Scored pilot audit" { & $Python scripts/audit_run.py --run $PilotScored --stage scored --require-revision }
    Run-Step "Automated pilot quality gate" { & $Python scripts/pilot_quality_gate.py --raw $PilotRaw --scored $PilotScored --output results/llada_gsm8k_pilot50_gate.json }
    Run-Step "Export 10-example manual review sheet" { & $Python scripts/export_trace_review.py --scored $PilotScored --output results/llada_gsm8k_pilot50_trace_review.md --count 10 }

    Write-Host ""
    Write-Host "AUTOMATED PILOT COMPLETE." -ForegroundColor Green
    Write-Host "Manual gate remains: review results/llada_gsm8k_pilot50_trace_review.md."
}
