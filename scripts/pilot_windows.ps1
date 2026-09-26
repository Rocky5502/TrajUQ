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

$ModelId = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"
$DatasetId = "openai/gsm8k"
$ModelConfigTemplate = "configs/models/llada.yaml"
$ModelConfig = "results/llada_pilot_model_config.yaml"
$ModelRevisionJson = "results/llada_model_revision.json"
$DatasetRevisionJson = "results/gsm8k_dataset_revision.json"
$CalData = "data/processed/gsm8k/calibration.jsonl"
$TestData = "data/processed/gsm8k/test.jsonl"
$ParityReference = "runs/llada_parity_reference.jsonl"
$ParityTrajUQ = "runs/llada_parity_trajuq.raw.jsonl"
$ParityResult = "results/llada_parity.json"
$PilotRaw = "runs/llada_gsm8k_pilot50.raw.jsonl"
$PilotScored = "runs/llada_gsm8k_pilot50.scored.jsonl"

New-Item -ItemType Directory -Force -Path results | Out-Null
New-Item -ItemType Directory -Force -Path runs | Out-Null

if ($Stage -in @("prepare", "all")) {
    Run-Step "Resolve model revision" { & $Python scripts/resolve_hf_revision.py --repo-type model --repo-id $ModelId --output $ModelRevisionJson }
    Run-Step "Resolve GSM8K revision" { & $Python scripts/resolve_hf_revision.py --repo-type dataset --repo-id $DatasetId --output $DatasetRevisionJson }

    $ModelSha = (Get-Content $ModelRevisionJson -Raw | ConvertFrom-Json).resolved_sha
    $DatasetSha = (Get-Content $DatasetRevisionJson -Raw | ConvertFrom-Json).resolved_sha
    if (-not $ModelSha) { throw "Model revision resolver returned no SHA." }
    if (-not $DatasetSha) { throw "Dataset revision resolver returned no SHA." }

    Write-Host "MODEL_SHA=$ModelSha"
    Write-Host "DATASET_SHA=$DatasetSha"

    $yaml = Get-Content $ModelConfigTemplate -Raw
    if ($yaml -notmatch '(?m)^revision:\s*.*$') { throw "Model config template has no revision field: $ModelConfigTemplate" }
    $yaml = $yaml -replace '(?m)^revision:\s*.*$', "revision: $ModelSha"
    Set-Content -Path $ModelConfig -Value $yaml -Encoding UTF8
    Write-Host "Frozen pilot model config -> $ModelConfig"

    Run-Step "Prepare frozen GSM8K pair" { & $Python scripts/prepare_eval_pair.py --dataset gsm8k --revision $DatasetSha }
    Run-Step "CUDA preflight" { & $Python scripts/preflight.py --model-config $ModelConfig --data $CalData --require-cuda }
    Run-Step "Environment snapshot" { & $Python scripts/environment_snapshot.py --output results/pilot_environment.json }
    Run-Step "Freeze manifest" { & $Python scripts/freeze_manifest.py --model-config $ModelConfig --dataset $CalData --dataset $TestData --output results/pilot_freeze_manifest.json }
}

$ModelSha = $null
if (Test-Path $ModelRevisionJson) {
    $ModelSha = (Get-Content $ModelRevisionJson -Raw | ConvertFrom-Json).resolved_sha
}
if (-not $ModelSha -and (Test-Path $ModelConfig)) {
    $match = Select-String -Path $ModelConfig -Pattern '^revision:\s*(\S+)\s*$'
    if ($match) { $ModelSha = $match.Matches[0].Groups[1].Value }
}
if (-not $ModelSha -or $ModelSha -eq "null") { throw "Exact model SHA is not frozen. Run -Stage prepare first." }

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
    Run-Step "Exact token parity" { & $Python scripts/compare_run_parity.py --reference $ParityReference --instrumented $ParityTrajUQ --require-token-ids --output $ParityResult }

    $Parity = Get-Content $ParityResult -Raw | ConvertFrom-Json
    if ($Parity.status -ne "PASS") { throw "Parity is not PASS. STOP: do not run the 50-item pilot." }

    Run-Step "Instrumentation overhead" { & $Python scripts/compare_instrumentation_overhead.py --reference $ParityReference --instrumented $ParityTrajUQ --output results/llada_instrumentation_overhead.json }
}

if ($Stage -in @("pilot", "all")) {
    if (-not (Test-Path $ParityResult)) { throw "Missing $ParityResult. Run -Stage parity first." }
    $Parity = Get-Content $ParityResult -Raw | ConvertFrom-Json
    if ($Parity.status -ne "PASS") { throw "Parity is not PASS. STOP." }

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
