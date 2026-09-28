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

function Get-FileSha256 {
    param([string]$Path)
    return (Get-FileHash -Algorithm SHA256 -Path $Path).Hash.ToLowerInvariant()
}
function Test-RunAuditPass {
    param(
        [string]$Artifact,
        [string]$ExpectedStage,
        [string]$ExpectedCommit
    )
    $AuditPath = "$Artifact.audit.json"
    if (-not (Test-Path $Artifact) -or -not (Test-Path $AuditPath)) {
        return $false
    }
    $Audit = Get-Content $AuditPath -Raw | ConvertFrom-Json
    return (
        $Audit.protocol -eq "trajuq_run_audit_v1" -and
        $Audit.status -eq "PASS" -and
        $Audit.git_clean -eq $true -and
        $Audit.git_commit -eq $ExpectedCommit -and
        $Audit.stage -eq $ExpectedStage -and
        $Audit.sha256 -eq (Get-FileSha256 $Artifact)
    )
}

function Require-No-ExistingArtifacts {
    param([string[]]$Paths)
    $Existing = @($Paths | Where-Object { Test-Path $_ })
    if ($Existing.Count -gt 0) {
        throw (
            "Existing partial/frozen artifacts require inspection before rerun: " +
            ($Existing -join ", ") +
            ". This wrapper will not delete or overwrite evidence."
        )
    }
}

function Assert-ParityPass {
    param(
        [string]$Path,
        [string]$ModelSha,
        [string]$Commit
    )
    if (-not (Test-Path $Path)) { return $false }
    $Parity = Get-Content $Path -Raw | ConvertFrom-Json
    $Reference = [string]$Parity.reference
    $Instrumented = [string]$Parity.instrumented
    $ReferenceHashOk = (
        (Test-Path $Reference) -and
        $Parity.reference_sha256 -eq (Get-FileSha256 $Reference)
    )
    $InstrumentedHashOk = (
        (Test-Path $Instrumented) -and
        $Parity.instrumented_sha256 -eq (Get-FileSha256 $Instrumented)
    )
    $InstrumentedAuditOk = Test-RunAuditPass $Instrumented "raw" $Commit
    $Pass = (
        $Parity.protocol -eq "trajuq_exact_parity_v3" -and
        $Parity.status -eq "PASS" -and
        $Parity.git_clean -eq $true -and
        $Parity.require_token_ids -eq $true -and
        $Parity.require_provenance -eq $true -and
        $Parity.git_commit -eq $Commit -and
        $Parity.shared_records -eq 20 -and
        $Parity.mismatch_count -eq 0 -and
        $Parity.provenance_mismatch_count -eq 0 -and
        $Parity.run_provenance.adapter -eq "llada" -and
        $Parity.run_provenance.model_revision -eq $ModelSha -and
        $Parity.run_provenance.source_git_commit -eq $Commit -and
        $Parity.run_provenance.source_git_clean -eq $true -and
        $ReferenceHashOk -and
        $InstrumentedHashOk -and
        $InstrumentedAuditOk
    )
    if (-not $Pass) {
        throw "Existing $Path is not a clean hash/audit-bound frozen parity-v3 PASS. Inspect it before rerunning."
    }
    return $true
}

function Assert-PilotGatePass {
    param(
        [string]$Path,
        [string]$Raw,
        [string]$Scored,
        [string]$ModelSha,
        [string]$Commit
    )
    if (-not (Test-Path $Path)) { return $false }
    if (-not (Test-Path $Raw) -or -not (Test-Path $Scored)) {
        throw "Pilot gate exists but its bound raw/scored artifacts are missing."
    }
    $Gate = Get-Content $Path -Raw | ConvertFrom-Json
    $Revisions = @($Gate.model_revisions)
    $RawAuditOk = Test-RunAuditPass $Raw "raw" $Commit
    $ScoredAuditOk = Test-RunAuditPass $Scored "scored" $Commit
    $Pass = (
        $Gate.protocol -eq "trajuq_pilot_quality_gate_v2" -and
        $Gate.status -eq "PASS" -and
        $Gate.git_clean -eq $true -and
        $Gate.git_commit -eq $Commit -and
        $Gate.raw -eq $Raw -and
        $Gate.scored -eq $Scored -and
        $Gate.raw_sha256 -eq (Get-FileSha256 $Raw) -and
        $Gate.scored_sha256 -eq (Get-FileSha256 $Scored) -and
        $RawAuditOk -and
        $ScoredAuditOk -and
        $Gate.raw_records -eq 50 -and
        $Gate.scored_records -eq 50 -and
        @($Gate.adapters).Count -eq 1 -and
        @($Gate.adapters)[0] -eq "llada" -and
        $Revisions.Count -eq 1 -and
        [string]$Revisions[0] -eq $ModelSha
    )
    if (-not $Pass) {
        throw "Existing $Path is not a clean hash/audit-bound frozen 50-example pilot PASS. Inspect it before rerunning."
    }
    return $true
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

    $Dirty = (& git status --porcelain) -join "`n"
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inspect git worktree."
    }
    if ($Dirty) {
        throw "Worktree is dirty after the pre-primary freeze. Commit/stash changes and re-run scripts/freeze_preprimary_windows.ps1 before GPU execution."
    }

    $CpuPath = "results/cpu_verification.json"
    if (-not (Test-Path $CpuPath)) {
        throw "Missing $CpuPath. Re-run scripts/freeze_preprimary_windows.ps1 before GPU execution."
    }
    $Cpu = Get-Content $CpuPath -Raw | ConvertFrom-Json
    if ($Cpu.protocol -ne "trajuq_cpu_verification_v1") {
        throw "Unexpected CPU-verification protocol: $($Cpu.protocol)"
    }
    if ($Cpu.status -ne "PASS" -or $Cpu.git_clean -ne $true) {
        throw "CPU verification is not a clean PASS."
    }
    if ($Cpu.git_commit -ne $Head) {
        throw "CPU verification commit $($Cpu.git_commit) does not match frozen HEAD $Head."
    }

    $GatedPath = "results/gated_dataset_access.json"
    if (-not (Test-Path $GatedPath)) {
        throw "Missing $GatedPath. Re-run scripts/freeze_preprimary_windows.ps1 before GPU execution."
    }
    $Gated = Get-Content $GatedPath -Raw | ConvertFrom-Json
    $LedgerHash = (Get-FileHash $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if (
        $Gated.protocol -ne "trajuq_gated_dataset_access_v1" -or
        $Gated.status -ne "PASS" -or
        $Gated.git_clean -ne $true -or
        $Gated.git_commit -ne $Head -or
        $Gated.revision_ledger_sha256 -ne $LedgerHash
    ) {
        throw "Gated dataset access is not a clean revision-bound PASS at frozen HEAD."
    }

    return $Ledger
}

function Assert-FrozenPilotInputs {
    param(
        [string]$CalibrationData,
        [string]$TestData,
        [string]$SplitManifestPath,
        [string]$SplitAuditPath,
        [string]$FreezeManifestPath,
        [string]$ModelConfigPath,
        [string]$DatasetSha,
        [string]$Commit
    )

    foreach ($Path in @(
        $CalibrationData,
        $TestData,
        $SplitManifestPath,
        $SplitAuditPath,
        $FreezeManifestPath,
        $ModelConfigPath
    )) {
        if (-not (Test-Path $Path)) {
            throw "Missing frozen pilot input: $Path. Run -Stage prepare first."
        }
    }

    $Split = Get-Content $SplitManifestPath -Raw | ConvertFrom-Json
    $Audit = Get-Content $SplitAuditPath -Raw | ConvertFrom-Json
    if ($Split.requested_revision -ne $DatasetSha) {
        throw "Frozen GSM8K split revision differs from the revision ledger."
    }
    if ($Split.source_git_clean -ne $true) {
        throw "Frozen GSM8K split was not produced from a clean checkout."
    }
    if ([string]$Split.source_git_commit -ne $Commit) {
        throw "Frozen GSM8K split source commit differs from the frozen experiment commit."
    }
    if ($Audit.source_git_clean -ne $true) {
        throw "Frozen GSM8K split audit does not attest a clean source checkout."
    }
    if ([string]$Audit.source_git_commit -ne $Commit) {
        throw "Frozen GSM8K split audit source commit differs from the frozen experiment commit."
    }
    if ($Split.calibration_sha256 -ne (Get-FileSha256 $CalibrationData)) {
        throw "Frozen GSM8K calibration hash differs from its split manifest."
    }
    if ($Split.test_sha256 -ne (Get-FileSha256 $TestData)) {
        throw "Frozen GSM8K test hash differs from its split manifest."
    }
    if (
        $Audit.protocol -ne "trajuq_split_pair_audit_v1" -or
        $Audit.status -ne "PASS" -or
        $Audit.requested_revision -ne $DatasetSha -or
        $Audit.calibration_sha256 -ne (Get-FileSha256 $CalibrationData) -or
        $Audit.test_sha256 -ne (Get-FileSha256 $TestData) -or
        $Audit.id_overlap_count -ne 0 -or
        $Audit.exact_prompt_overlap_count -ne 0
    ) {
        throw "Frozen GSM8K split audit is stale or not a zero-overlap PASS."
    }

    $Freeze = Get-Content $FreezeManifestPath -Raw | ConvertFrom-Json
    if ($Freeze.git_commit -ne $Commit) {
        throw "Pilot freeze manifest belongs to another git commit."
    }
    if ($Freeze.model_config_sha256 -ne (Get-FileSha256 $ModelConfigPath)) {
        throw "Pilot freeze manifest model-config hash drifted."
    }
    $FrozenDatasets = @{}
    foreach ($Item in @($Freeze.datasets)) {
        $FrozenDatasets[[string]$Item.path] = [string]$Item.sha256
    }
    if (-not $FrozenDatasets.ContainsKey($CalibrationData)) {
        throw "Pilot freeze manifest is missing the calibration dataset path."
    }
    if (-not $FrozenDatasets.ContainsKey($TestData)) {
        throw "Pilot freeze manifest is missing the test dataset path."
    }
    if ($FrozenDatasets[$CalibrationData] -ne (Get-FileSha256 $CalibrationData)) {
        throw "Pilot freeze manifest calibration hash drifted."
    }
    if ($FrozenDatasets[$TestData] -ne (Get-FileSha256 $TestData)) {
        throw "Pilot freeze manifest test hash drifted."
    }
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
$PilotGate = "results/llada_gsm8k_pilot50_gate.json"
$PilotReview = "results/llada_gsm8k_pilot50_trace_review.md"
$ParityOverhead = "results/llada_instrumentation_overhead.json"
$SplitManifest = "data/processed/gsm8k/split_manifest.json"
$SplitAudit = "data/processed/gsm8k/split_audit.json"

New-Item -ItemType Directory -Force -Path results | Out-Null
New-Item -ItemType Directory -Force -Path runs | Out-Null

Run-Step "Validate frozen pre-primary evidence" {
    & $Python scripts/validate_frozen_preprimary.py
}

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
    $SplitArtifacts = @($CalData, $TestData, $SplitManifest, $SplitAudit)
    $ExistingSplitArtifacts = @($SplitArtifacts | Where-Object { Test-Path $_ })
    if ($ExistingSplitArtifacts.Count -eq 0) {
        Run-Step "Prepare frozen GSM8K pair" {
            & $Python scripts/prepare_eval_pair.py --dataset gsm8k --revision $DatasetSha
        }
    }
    elseif ($ExistingSplitArtifacts.Count -ne $SplitArtifacts.Count) {
        throw (
            "Partial GSM8K split artifacts exist: " +
            ($ExistingSplitArtifacts -join ", ") +
            ". Inspect them before rerunning; this wrapper will not overwrite frozen data."
        )
    }
    else {
        $Split = Get-Content $SplitManifest -Raw | ConvertFrom-Json
        $Audit = Get-Content $SplitAudit -Raw | ConvertFrom-Json
        if ($Split.requested_revision -ne $DatasetSha) {
            throw "Existing GSM8K split revision differs from the frozen revision ledger."
        }
        if ($Split.calibration_sha256 -ne (Get-FileSha256 $CalData)) {
            throw "Existing GSM8K calibration hash differs from its split manifest."
        }
        if ($Split.test_sha256 -ne (Get-FileSha256 $TestData)) {
            throw "Existing GSM8K test hash differs from its split manifest."
        }
        if (
            $Audit.status -ne "PASS" -or
            $Audit.id_overlap_count -ne 0 -or
            $Audit.exact_prompt_overlap_count -ne 0
        ) {
            throw "Existing GSM8K split audit is not a zero-overlap PASS."
        }
        Write-Host "Frozen GSM8K pair already present and hash-valid; preserving it." -ForegroundColor Green
    }

    Run-Step "CUDA preflight" {
        & $Python scripts/preflight.py --model-config $ModelConfig --data $CalData --require-cuda
    }

    if (-not (Test-Path "results/pilot_environment.json")) {
        Run-Step "Environment snapshot" {
            & $Python scripts/environment_snapshot.py --output results/pilot_environment.json
        }
    }
    else {
        Write-Host "Preserving existing results/pilot_environment.json." -ForegroundColor Green
    }

    if (-not (Test-Path "results/pilot_freeze_manifest.json")) {
        Run-Step "Freeze manifest" {
            & $Python scripts/freeze_manifest.py --model-config $ModelConfig --dataset $CalData --dataset $TestData --output results/pilot_freeze_manifest.json
        }
    }
    else {
        $Freeze = Get-Content results/pilot_freeze_manifest.json -Raw | ConvertFrom-Json
        if ($Freeze.git_commit -ne $Ledger.git_commit) {
            throw "Existing pilot freeze manifest belongs to another git commit."
        }
        if ($Freeze.model_config_sha256 -ne (Get-FileSha256 $ModelConfig)) {
            throw "Existing pilot freeze manifest model-config hash drifted."
        }
        $FrozenDatasets = @{}
        foreach ($Item in @($Freeze.datasets)) {
            $FrozenDatasets[[string]$Item.path] = [string]$Item.sha256
        }
        if (-not $FrozenDatasets.ContainsKey($CalData)) {
            throw "Existing pilot freeze manifest is missing the calibration dataset path."
        }
        if (-not $FrozenDatasets.ContainsKey($TestData)) {
            throw "Existing pilot freeze manifest is missing the test dataset path."
        }
        if ($FrozenDatasets[$CalData] -ne (Get-FileSha256 $CalData)) {
            throw "Existing pilot freeze manifest calibration hash drifted."
        }
        if ($FrozenDatasets[$TestData] -ne (Get-FileSha256 $TestData)) {
            throw "Existing pilot freeze manifest test hash drifted."
        }
        Write-Host "Preserving existing results/pilot_freeze_manifest.json." -ForegroundColor Green
    }
}

if ($Stage -in @("parity", "pilot", "all")) {
    Assert-FrozenPilotInputs `
        -CalibrationData $CalData `
        -TestData $TestData `
        -SplitManifestPath $SplitManifest `
        -SplitAuditPath $SplitAudit `
        -FreezeManifestPath "results/pilot_freeze_manifest.json" `
        -ModelConfigPath $ModelConfig `
        -DatasetSha $DatasetSha `
        -Commit $Ledger.git_commit
}
if ($Stage -in @("parity", "all")) {
    if (Assert-ParityPass $ParityResult $ModelSha $Ledger.git_commit) {
        Write-Host "LLaDA parity already PASS and hash-valid; preserving existing evidence." -ForegroundColor Green
        Run-Step "Validate instrumentation overhead" {
            & $Python scripts/compare_instrumentation_overhead.py --reference $ParityReference --instrumented $ParityTrajUQ --output $ParityOverhead
        }
    }
    else {
        Require-No-ExistingArtifacts @(
            $ParityReference,
            $ParityTrajUQ,
            "$ParityTrajUQ.audit.json",
            $ParityOverhead
        )

        Run-Step "20-prompt native reference" {
            & $Python scripts/run_llada_reference.py --model $ModelId --revision $ModelSha --input $CalData --output $ParityReference --limit 20 --steps 128 --max-new-tokens 128 --block-length 128 --temperature 0
        }
        Run-Step "20-prompt TrajUQ telemetry" {
            & $Python scripts/run_generation.py --adapter llada --model $ModelId --revision $ModelSha --input $CalData --output $ParityTrajUQ --limit 20 --steps 128 --max-new-tokens 128 --block-length 128 --temperature 0
        }
        Run-Step "Raw parity-run audit" {
            & $Python scripts/audit_run.py --run $ParityTrajUQ --stage raw --require-revision
        }
        Run-Step "Exact token parity" {
            & $Python scripts/compare_run_parity.py --reference $ParityReference --instrumented $ParityTrajUQ --require-token-ids --require-provenance --output $ParityResult
        }

        if (-not (Assert-ParityPass $ParityResult $ModelSha $Ledger.git_commit)) {
            throw "Parity did not produce a reusable PASS artifact."
        }

        Run-Step "Instrumentation overhead" {
            & $Python scripts/compare_instrumentation_overhead.py --reference $ParityReference --instrumented $ParityTrajUQ --output $ParityOverhead
        }
    }
}

if ($Stage -in @("pilot", "all")) {
    if (-not (Assert-ParityPass $ParityResult $ModelSha $Ledger.git_commit)) {
        throw "Missing valid LLaDA parity PASS. Run -Stage parity first."
    }

    if (Assert-PilotGatePass $PilotGate $PilotRaw $PilotScored $ModelSha $Ledger.git_commit) {
        Write-Host "LLaDA + GSM8K 50-example pilot already PASS and hash-valid; preserving evidence." -ForegroundColor Green
        if (-not (Test-Path $PilotReview)) {
            Run-Step "Export 10-example manual review sheet" {
                & $Python scripts/export_trace_review.py --scored $PilotScored --output $PilotReview --count 10
            }
        }
    }
    else {
        Require-No-ExistingArtifacts @(
            $PilotRaw,
            "$PilotRaw.audit.json",
            $PilotScored,
            "$PilotScored.audit.json",
            $PilotReview,
            "results/llada_gsm8k_pilot50_manual_review.json"
        )

        Run-Step "50-item TrajUQ pilot" {
            & $Python scripts/run_generation.py --adapter llada --model $ModelId --revision $ModelSha --input $CalData --output $PilotRaw --limit 50 --steps 128 --max-new-tokens 128 --block-length 128 --temperature 0
        }
        Run-Step "Raw pilot audit" {
            & $Python scripts/audit_run.py --run $PilotRaw --stage raw --require-revision
        }
        Run-Step "Score pilot into separate artifact" {
            & $Python scripts/score_outputs.py --run $PilotRaw --input $CalData --output $PilotScored
        }
        Run-Step "Scored pilot audit" {
            & $Python scripts/audit_run.py --run $PilotScored --stage scored --require-revision
        }
        Run-Step "Automated pilot quality gate" {
            & $Python scripts/pilot_quality_gate.py --raw $PilotRaw --scored $PilotScored --output $PilotGate
        }
        if (-not (Assert-PilotGatePass $PilotGate $PilotRaw $PilotScored $ModelSha $Ledger.git_commit)) {
            throw "Pilot quality gate did not produce a reusable PASS artifact."
        }
        Run-Step "Export 10-example manual review sheet" {
            & $Python scripts/export_trace_review.py --scored $PilotScored --output $PilotReview --count 10
        }
    }

    Write-Host ""
    Write-Host "AUTOMATED PILOT COMPLETE." -ForegroundColor Green
    Write-Host "Manual gate remains: review $PilotReview."
}
