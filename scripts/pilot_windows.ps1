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

$ModelId = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"
$DatasetId = "openai/gsm8k"
$ModelConfigTemplate = "configs/models/llada.yaml"
$ModelConfig = "results/llada_pilot_model_config.yaml"
$CalData = "data/processed/gsm8k/calibration.jsonl"
$TestData = "data/processed/gsm8k/test.jsonl"

New-Item -ItemType Directory -Force -Path results | Out-Null
New-Item -ItemType Directory -Force -Path runs | Out-Null

if ($Stage -in @("prepare","all")) {
    Run-Step "Resolve model revision" { & $Python scripts/resolve_hf_revision.py --repo-type model --repo-id $ModelId --output results/llada_model_revision.json }
    Run-Step "Resolve GSM8K revision" { & $Python scripts/resolve_hf_revision.py --repo-type dataset --repo-id $DatasetId --output results/gsm8k_dataset_revision.json }

    $ModelSha = (Get-Content results/llada_model_revision.json | ConvertFrom-Json).resolved_sha
    $DatasetSha = (Get-Content results/gsm8k_dataset_revision.json | ConvertFrom-Json).resolved_sha
    Write-Host "MODEL_SHA=$ModelSha"
    Write-Host "DATASET_SHA=$DatasetSha"

    $yaml = Get-Content $ModelConfigTemplate -Raw
    $yaml = $yaml -replace '(?m)^revision:\s*.*

    Run-Step "Prepare frozen GSM8K pair" { & $Python scripts/prepare_eval_pair.py --dataset gsm8k --revision $DatasetSha }
    Run-Step "CUDA preflight" { & $Python scripts/preflight.py --model-config $ModelConfig --data $CalData --require-cuda }
    Run-Step "Environment snapshot" { & $Python scripts/environment_snapshot.py --output results/pilot_environment.json }
    Run-Step "Freeze manifest" { & $Python scripts/freeze_manifest.py --model-config $ModelConfig --dataset $CalData --dataset $TestData --output results/pilot_freeze_manifest.json }
}

$ModelSha = $null
if (Test-Path results/llada_model_revision.json) {
    $ModelSha = (Get-Content results/llada_model_revision.json | ConvertFrom-Json).resolved_sha
}
if (-not $ModelSha -and (Test-Path $ModelConfig)) {
    $match = Select-String -Path $ModelConfig -Pattern '^revision:\s*(\S+)\s*
    if ($match) { $ModelSha = $match.Matches[0].Groups[1].Value }
}
if (-not $ModelSha -or $ModelSha -eq "null") { throw "Exact model SHA is not frozen. Run -Stage prepare first." }

if ($Stage -in @("parity","all")) {
    Remove-Item runs/llada_parity_reference.jsonl -ErrorAction SilentlyContinue
    Remove-Item runs/llada_parity_trajuq.raw.jsonl -ErrorAction SilentlyContinue

    Run-Step "20-prompt native reference" { & $Python scripts/run_llada_reference.py --model $ModelId --revision $ModelSha --input $CalData --output runs/llada_parity_reference.jsonl --limit 20 --steps 128 --max-new-tokens 128 --block-length 128 --temperature 0 }
    Run-Step "20-prompt TrajUQ telemetry" { & $Python scripts/run_generation.py --adapter llada --model $ModelId --revision $ModelSha --input $CalData --output runs/llada_parity_trajuq.raw.jsonl --limit 20 --steps 128 --max-new-tokens 128 --block-length 128 }
    Run-Step "Raw parity-run audit" { & $Python scripts/audit_run.py --run runs/llada_parity_trajuq.raw.jsonl --stage raw --require-revision }
    Run-Step "Exact token parity" { & $Python scripts/compare_run_parity.py --reference runs/llada_parity_reference.jsonl --instrumented runs/llada_parity_trajuq.raw.jsonl --require-token-ids --output results/llada_parity.json }

    $Parity = Get-Content results/llada_parity.json | ConvertFrom-Json
    if ($Parity.status -ne "PASS") { throw "Parity is not PASS. STOP: do not run the 50-item pilot." }

    Run-Step "Instrumentation overhead" { & $Python scripts/compare_instrumentation_overhead.py --reference runs/llada_parity_reference.jsonl --instrumented runs/llada_parity_trajuq.raw.jsonl --output results/llada_instrumentation_overhead.json }
}

if ($Stage -in @("pilot","all")) {
    if (-not (Test-Path results/llada_parity.json)) { throw "Missing results/llada_parity.json. Run -Stage parity first." }
    $Parity = Get-Content results/llada_parity.json | ConvertFrom-Json
    if ($Parity.status -ne "PASS") { throw "Parity is not PASS. STOP." }

    Remove-Item runs/llada_gsm8k_pilot50.raw.jsonl -ErrorAction SilentlyContinue
    Remove-Item runs/llada_gsm8k_pilot50.scored.jsonl -ErrorAction SilentlyContinue

    Run-Step "50-item TrajUQ pilot" { & $Python scripts/run_generation.py --adapter llada --model $ModelId --revision $ModelSha --input $CalData --output runs/llada_gsm8k_pilot50.raw.jsonl --limit 50 --steps 128 --max-new-tokens 128 --block-length 128 }
    Run-Step "Raw pilot audit" { & $Python scripts/audit_run.py --run runs/llada_gsm8k_pilot50.raw.jsonl --stage raw --require-revision }
    Run-Step "Score pilot into separate artifact" { & $Python scripts/score_outputs.py --run runs/llada_gsm8k_pilot50.raw.jsonl --input $CalData --output runs/llada_gsm8k_pilot50.scored.jsonl }
    Run-Step "Scored pilot audit" { & $Python scripts/audit_run.py --run runs/llada_gsm8k_pilot50.scored.jsonl --stage scored --require-revision }
    Run-Step "Automated pilot quality gate" { & $Python scripts/pilot_quality_gate.py --raw runs/llada_gsm8k_pilot50.raw.jsonl --scored runs/llada_gsm8k_pilot50.scored.jsonl --output results/llada_gsm8k_pilot50_gate.json }
    Run-Step "Export 10-example manual review sheet" { & $Python scripts/export_trace_review.py --scored runs/llada_gsm8k_pilot50.scored.jsonl --output results/llada_gsm8k_pilot50_trace_review.md --count 10 }

    Write-Host ""
    Write-Host "AUTOMATED PILOT COMPLETE." -ForegroundColor Green
    Write-Host "Manual gate remains: review results/llada_gsm8k_pilot50_trace_review.md."
}
, "revision: $ModelSha"
    Set-Content -Path $ModelConfig -Value $yaml -Encoding UTF8
    Write-Host "Frozen pilot model config -> $ModelConfig"

    Run-Step "Prepare frozen GSM8K pair" { & $Python scripts/prepare_eval_pair.py --dataset gsm8k --revision $DatasetSha }
    Run-Step "CUDA preflight" { & $Python scripts/preflight.py --model-config $ModelConfig --data $CalData --require-cuda }
    Run-Step "Environment snapshot" { & $Python scripts/environment_snapshot.py --output results/pilot_environment.json }
    Run-Step "Freeze manifest" { & $Python scripts/freeze_manifest.py --model-config $ModelConfig --dataset $CalData --dataset $TestData --output results/pilot_freeze_manifest.json }
}

$ModelSha = $null
if (Test-Path results/llada_model_revision.json) {
    $ModelSha = (Get-Content results/llada_model_revision.json | ConvertFrom-Json).resolved_sha
}
if (-not $ModelSha) {
    $match = Select-String -Path $ModelConfig -Pattern '^revision:\s*(\S+)\s*$'
    if ($match) { $ModelSha = $match.Matches[0].Groups[1].Value }
}
if (-not $ModelSha -or $ModelSha -eq "null") { throw "Exact model SHA is not frozen. Run -Stage prepare first." }

if ($Stage -in @("parity","all")) {
    Remove-Item runs/llada_parity_reference.jsonl -ErrorAction SilentlyContinue
    Remove-Item runs/llada_parity_trajuq.raw.jsonl -ErrorAction SilentlyContinue

    Run-Step "20-prompt native reference" { & $Python scripts/run_llada_reference.py --model $ModelId --revision $ModelSha --input $CalData --output runs/llada_parity_reference.jsonl --limit 20 --steps 128 --max-new-tokens 128 --block-length 128 --temperature 0 }
    Run-Step "20-prompt TrajUQ telemetry" { & $Python scripts/run_generation.py --adapter llada --model $ModelId --revision $ModelSha --input $CalData --output runs/llada_parity_trajuq.raw.jsonl --limit 20 --steps 128 --max-new-tokens 128 --block-length 128 }
    Run-Step "Raw parity-run audit" { & $Python scripts/audit_run.py --run runs/llada_parity_trajuq.raw.jsonl --stage raw --require-revision }
    Run-Step "Exact token parity" { & $Python scripts/compare_run_parity.py --reference runs/llada_parity_reference.jsonl --instrumented runs/llada_parity_trajuq.raw.jsonl --require-token-ids --output results/llada_parity.json }

    $Parity = Get-Content results/llada_parity.json | ConvertFrom-Json
    if ($Parity.status -ne "PASS") { throw "Parity is not PASS. STOP: do not run the 50-item pilot." }

    Run-Step "Instrumentation overhead" { & $Python scripts/compare_instrumentation_overhead.py --reference runs/llada_parity_reference.jsonl --instrumented runs/llada_parity_trajuq.raw.jsonl --output results/llada_instrumentation_overhead.json }
}

if ($Stage -in @("pilot","all")) {
    if (-not (Test-Path results/llada_parity.json)) { throw "Missing results/llada_parity.json. Run -Stage parity first." }
    $Parity = Get-Content results/llada_parity.json | ConvertFrom-Json
    if ($Parity.status -ne "PASS") { throw "Parity is not PASS. STOP." }

    Remove-Item runs/llada_gsm8k_pilot50.raw.jsonl -ErrorAction SilentlyContinue
    Remove-Item runs/llada_gsm8k_pilot50.scored.jsonl -ErrorAction SilentlyContinue

    Run-Step "50-item TrajUQ pilot" { & $Python scripts/run_generation.py --adapter llada --model $ModelId --revision $ModelSha --input $CalData --output runs/llada_gsm8k_pilot50.raw.jsonl --limit 50 --steps 128 --max-new-tokens 128 --block-length 128 }
    Run-Step "Raw pilot audit" { & $Python scripts/audit_run.py --run runs/llada_gsm8k_pilot50.raw.jsonl --stage raw --require-revision }
    Run-Step "Score pilot into separate artifact" { & $Python scripts/score_outputs.py --run runs/llada_gsm8k_pilot50.raw.jsonl --input $CalData --output runs/llada_gsm8k_pilot50.scored.jsonl }
    Run-Step "Scored pilot audit" { & $Python scripts/audit_run.py --run runs/llada_gsm8k_pilot50.scored.jsonl --stage scored --require-revision }
    Run-Step "Automated pilot quality gate" { & $Python scripts/pilot_quality_gate.py --raw runs/llada_gsm8k_pilot50.raw.jsonl --scored runs/llada_gsm8k_pilot50.scored.jsonl --output results/llada_gsm8k_pilot50_gate.json }
    Run-Step "Export 10-example manual review sheet" { & $Python scripts/export_trace_review.py --scored runs/llada_gsm8k_pilot50.scored.jsonl --output results/llada_gsm8k_pilot50_trace_review.md --count 10 }

    Write-Host ""
    Write-Host "AUTOMATED PILOT COMPLETE." -ForegroundColor Green
    Write-Host "Manual gate remains: review results/llada_gsm8k_pilot50_trace_review.md."
}

    if ($match) { $ModelSha = $match.Matches[0].Groups[1].Value }
}
if (-not $ModelSha -or $ModelSha -eq "null") { throw "Exact model SHA is not frozen. Run -Stage prepare first." }

if ($Stage -in @("parity","all")) {
    Remove-Item runs/llada_parity_reference.jsonl -ErrorAction SilentlyContinue
    Remove-Item runs/llada_parity_trajuq.raw.jsonl -ErrorAction SilentlyContinue

    Run-Step "20-prompt native reference" { & $Python scripts/run_llada_reference.py --model $ModelId --revision $ModelSha --input $CalData --output runs/llada_parity_reference.jsonl --limit 20 --steps 128 --max-new-tokens 128 --block-length 128 --temperature 0 }
    Run-Step "20-prompt TrajUQ telemetry" { & $Python scripts/run_generation.py --adapter llada --model $ModelId --revision $ModelSha --input $CalData --output runs/llada_parity_trajuq.raw.jsonl --limit 20 --steps 128 --max-new-tokens 128 --block-length 128 }
    Run-Step "Raw parity-run audit" { & $Python scripts/audit_run.py --run runs/llada_parity_trajuq.raw.jsonl --stage raw --require-revision }
    Run-Step "Exact token parity" { & $Python scripts/compare_run_parity.py --reference runs/llada_parity_reference.jsonl --instrumented runs/llada_parity_trajuq.raw.jsonl --require-token-ids --output results/llada_parity.json }

    $Parity = Get-Content results/llada_parity.json | ConvertFrom-Json
    if ($Parity.status -ne "PASS") { throw "Parity is not PASS. STOP: do not run the 50-item pilot." }

    Run-Step "Instrumentation overhead" { & $Python scripts/compare_instrumentation_overhead.py --reference runs/llada_parity_reference.jsonl --instrumented runs/llada_parity_trajuq.raw.jsonl --output results/llada_instrumentation_overhead.json }
}

if ($Stage -in @("pilot","all")) {
    if (-not (Test-Path results/llada_parity.json)) { throw "Missing results/llada_parity.json. Run -Stage parity first." }
    $Parity = Get-Content results/llada_parity.json | ConvertFrom-Json
    if ($Parity.status -ne "PASS") { throw "Parity is not PASS. STOP." }

    Remove-Item runs/llada_gsm8k_pilot50.raw.jsonl -ErrorAction SilentlyContinue
    Remove-Item runs/llada_gsm8k_pilot50.scored.jsonl -ErrorAction SilentlyContinue

    Run-Step "50-item TrajUQ pilot" { & $Python scripts/run_generation.py --adapter llada --model $ModelId --revision $ModelSha --input $CalData --output runs/llada_gsm8k_pilot50.raw.jsonl --limit 50 --steps 128 --max-new-tokens 128 --block-length 128 }
    Run-Step "Raw pilot audit" { & $Python scripts/audit_run.py --run runs/llada_gsm8k_pilot50.raw.jsonl --stage raw --require-revision }
    Run-Step "Score pilot into separate artifact" { & $Python scripts/score_outputs.py --run runs/llada_gsm8k_pilot50.raw.jsonl --input $CalData --output runs/llada_gsm8k_pilot50.scored.jsonl }
    Run-Step "Scored pilot audit" { & $Python scripts/audit_run.py --run runs/llada_gsm8k_pilot50.scored.jsonl --stage scored --require-revision }
    Run-Step "Automated pilot quality gate" { & $Python scripts/pilot_quality_gate.py --raw runs/llada_gsm8k_pilot50.raw.jsonl --scored runs/llada_gsm8k_pilot50.scored.jsonl --output results/llada_gsm8k_pilot50_gate.json }
    Run-Step "Export 10-example manual review sheet" { & $Python scripts/export_trace_review.py --scored runs/llada_gsm8k_pilot50.scored.jsonl --output results/llada_gsm8k_pilot50_trace_review.md --count 10 }

    Write-Host ""
    Write-Host "AUTOMATED PILOT COMPLETE." -ForegroundColor Green
    Write-Host "Manual gate remains: review results/llada_gsm8k_pilot50_trace_review.md."
}
, "revision: $ModelSha"
    Set-Content -Path $ModelConfig -Value $yaml -Encoding UTF8
    Write-Host "Frozen pilot model config -> $ModelConfig"

    Run-Step "Prepare frozen GSM8K pair" { & $Python scripts/prepare_eval_pair.py --dataset gsm8k --revision $DatasetSha }
    Run-Step "CUDA preflight" { & $Python scripts/preflight.py --model-config $ModelConfig --data $CalData --require-cuda }
    Run-Step "Environment snapshot" { & $Python scripts/environment_snapshot.py --output results/pilot_environment.json }
    Run-Step "Freeze manifest" { & $Python scripts/freeze_manifest.py --model-config $ModelConfig --dataset $CalData --dataset $TestData --output results/pilot_freeze_manifest.json }
}

$ModelSha = $null
if (Test-Path results/llada_model_revision.json) {
    $ModelSha = (Get-Content results/llada_model_revision.json | ConvertFrom-Json).resolved_sha
}
if (-not $ModelSha) {
    $match = Select-String -Path $ModelConfig -Pattern '^revision:\s*(\S+)\s*$'
    if ($match) { $ModelSha = $match.Matches[0].Groups[1].Value }
}
if (-not $ModelSha -or $ModelSha -eq "null") { throw "Exact model SHA is not frozen. Run -Stage prepare first." }

if ($Stage -in @("parity","all")) {
    Remove-Item runs/llada_parity_reference.jsonl -ErrorAction SilentlyContinue
    Remove-Item runs/llada_parity_trajuq.raw.jsonl -ErrorAction SilentlyContinue

    Run-Step "20-prompt native reference" { & $Python scripts/run_llada_reference.py --model $ModelId --revision $ModelSha --input $CalData --output runs/llada_parity_reference.jsonl --limit 20 --steps 128 --max-new-tokens 128 --block-length 128 --temperature 0 }
    Run-Step "20-prompt TrajUQ telemetry" { & $Python scripts/run_generation.py --adapter llada --model $ModelId --revision $ModelSha --input $CalData --output runs/llada_parity_trajuq.raw.jsonl --limit 20 --steps 128 --max-new-tokens 128 --block-length 128 }
    Run-Step "Raw parity-run audit" { & $Python scripts/audit_run.py --run runs/llada_parity_trajuq.raw.jsonl --stage raw --require-revision }
    Run-Step "Exact token parity" { & $Python scripts/compare_run_parity.py --reference runs/llada_parity_reference.jsonl --instrumented runs/llada_parity_trajuq.raw.jsonl --require-token-ids --output results/llada_parity.json }

    $Parity = Get-Content results/llada_parity.json | ConvertFrom-Json
    if ($Parity.status -ne "PASS") { throw "Parity is not PASS. STOP: do not run the 50-item pilot." }

    Run-Step "Instrumentation overhead" { & $Python scripts/compare_instrumentation_overhead.py --reference runs/llada_parity_reference.jsonl --instrumented runs/llada_parity_trajuq.raw.jsonl --output results/llada_instrumentation_overhead.json }
}

if ($Stage -in @("pilot","all")) {
    if (-not (Test-Path results/llada_parity.json)) { throw "Missing results/llada_parity.json. Run -Stage parity first." }
    $Parity = Get-Content results/llada_parity.json | ConvertFrom-Json
    if ($Parity.status -ne "PASS") { throw "Parity is not PASS. STOP." }

    Remove-Item runs/llada_gsm8k_pilot50.raw.jsonl -ErrorAction SilentlyContinue
    Remove-Item runs/llada_gsm8k_pilot50.scored.jsonl -ErrorAction SilentlyContinue

    Run-Step "50-item TrajUQ pilot" { & $Python scripts/run_generation.py --adapter llada --model $ModelId --revision $ModelSha --input $CalData --output runs/llada_gsm8k_pilot50.raw.jsonl --limit 50 --steps 128 --max-new-tokens 128 --block-length 128 }
    Run-Step "Raw pilot audit" { & $Python scripts/audit_run.py --run runs/llada_gsm8k_pilot50.raw.jsonl --stage raw --require-revision }
    Run-Step "Score pilot into separate artifact" { & $Python scripts/score_outputs.py --run runs/llada_gsm8k_pilot50.raw.jsonl --input $CalData --output runs/llada_gsm8k_pilot50.scored.jsonl }
    Run-Step "Scored pilot audit" { & $Python scripts/audit_run.py --run runs/llada_gsm8k_pilot50.scored.jsonl --stage scored --require-revision }
    Run-Step "Automated pilot quality gate" { & $Python scripts/pilot_quality_gate.py --raw runs/llada_gsm8k_pilot50.raw.jsonl --scored runs/llada_gsm8k_pilot50.scored.jsonl --output results/llada_gsm8k_pilot50_gate.json }
    Run-Step "Export 10-example manual review sheet" { & $Python scripts/export_trace_review.py --scored runs/llada_gsm8k_pilot50.scored.jsonl --output results/llada_gsm8k_pilot50_trace_review.md --count 10 }

    Write-Host ""
    Write-Host "AUTOMATED PILOT COMPLETE." -ForegroundColor Green
    Write-Host "Manual gate remains: review results/llada_gsm8k_pilot50_trace_review.md."
}
