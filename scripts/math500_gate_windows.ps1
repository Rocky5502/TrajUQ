param(
    [ValidateSet("prepare","pilot","all")]
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

function Read-FrozenState {
    $LedgerPath = "results/revision_ledger.json"
    $CpuPath = "results/cpu_verification.json"
    if (-not (Test-Path $LedgerPath)) {
        throw "Missing $LedgerPath. Run scripts/freeze_preprimary_windows.ps1 first."
    }
    if (-not (Test-Path $CpuPath)) {
        throw "Missing $CpuPath. Run scripts/freeze_preprimary_windows.ps1 first."
    }

    $Ledger = Get-Content $LedgerPath -Raw | ConvertFrom-Json
    $Cpu = Get-Content $CpuPath -Raw | ConvertFrom-Json
    $Head = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $Head) {
        throw "Unable to resolve current git commit."
    }
    $Dirty = (& git status --porcelain) -join "`n"
    if ($LASTEXITCODE -ne 0) { throw "Unable to inspect git worktree." }
    if ($Dirty) {
        throw "Worktree is dirty. Restore the frozen clean checkout before MATH-500 gating."
    }

    if ($Ledger.protocol -ne "trajuq_revision_ledger_v2" -or $Ledger.status -ne "PASS") {
        throw "Revision ledger is not a protocol-v2 PASS."
    }
    if (
        $Cpu.protocol -ne "trajuq_cpu_verification_v1" -or
        $Cpu.status -ne "PASS" -or
        $Cpu.git_clean -ne $true
    ) {
        throw "CPU verification is not a clean protocol-v1 PASS."
    }
    if ($Ledger.git_commit -ne $Head -or $Cpu.git_commit -ne $Head) {
        throw "Frozen CPU/revision artifacts do not match current HEAD $Head."
    }

    return @{
        Ledger = $Ledger
        Head = $Head
    }
}

function Require-LLaDAParity {
    param($Ledger, [string]$Head)

    $Entry = @($Ledger.models | Where-Object { $_.adapter -eq "llada" })
    if ($Entry.Count -ne 1) { throw "Revision ledger must contain exactly one LLaDA entry." }
    $Revision = [string]$Entry[0].resolved_sha
    $ParityPath = "results/llada_parity.json"
    if (-not (Test-Path $ParityPath)) {
        throw "Missing $ParityPath. LLaDA exact parity must PASS before the MATH-500 scorer pilot."
    }
    $Parity = Get-Content $ParityPath -Raw | ConvertFrom-Json
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
    $Pass = (
        $Parity.protocol -eq "trajuq_exact_parity_v3" -and
        $Parity.status -eq "PASS" -and
        $Parity.git_clean -eq $true -and
        $Parity.require_token_ids -eq $true -and
        $Parity.require_provenance -eq $true -and
        $Parity.git_commit -eq $Head -and
        $Parity.shared_records -eq 20 -and
        $Parity.mismatch_count -eq 0 -and
        $Parity.provenance_mismatch_count -eq 0 -and
        $Parity.run_provenance.adapter -eq "llada" -and
        $Parity.run_provenance.model_revision -eq $Revision -and
        $ReferenceHashOk -and
        $InstrumentedHashOk
    )
    if (-not $Pass) {
        throw "LLaDA parity is not an exact 20-prompt PASS at the frozen commit/revision."
    }
}

function Require-ScorerAudit {
    param([string]$Head)

    $AuditPath = "results/math500_scorer_audit.json"
    if (-not (Test-Path $AuditPath)) {
        Run-Step "MATH-500 frozen scorer audit" {
            & $Python scripts/audit_math500_scorer.py --output $AuditPath
        }
    }
    $Audit = Get-Content $AuditPath -Raw | ConvertFrom-Json
    $Pass = (
        $Audit.protocol -eq "trajuq_math500_scorer_audit_v2" -and
        $Audit.status -eq "PASS" -and
        $Audit.git_clean -eq $true -and
        $Audit.git_commit -eq $Head -and
        $Audit.failure_count -eq 0 -and
        $Audit.dependency_versions.'math-verify' -eq "0.9.0" -and
        $Audit.dependency_versions.'antlr4-python3-runtime' -eq "4.13.2"
    )
    if (-not $Pass) {
        throw "MATH-500 automatic scorer audit is stale, dependency-drifted, or not PASS."
    }
}

function Assert-Math500FrozenPair {
    param(
        [string]$CalibrationData,
        [string]$TestData,
        [string]$SplitManifestPath,
        [string]$SplitAuditPath,
        [string]$DatasetSha
    )

    foreach ($Path in @($CalibrationData, $TestData, $SplitManifestPath, $SplitAuditPath)) {
        if (-not (Test-Path $Path)) {
            throw "Missing frozen MATH-500 split artifact: $Path. Run -Stage prepare first."
        }
    }

    $Split = Get-Content $SplitManifestPath -Raw | ConvertFrom-Json
    $Audit = Get-Content $SplitAuditPath -Raw | ConvertFrom-Json
    if ($Split.requested_revision -ne $DatasetSha) {
        throw "Frozen MATH-500 split revision differs from the revision ledger."
    }
    if ($Split.calibration_sha256 -ne (Get-FileSha256 $CalibrationData)) {
        throw "Frozen MATH-500 calibration hash differs from its split manifest."
    }
    if ($Split.test_sha256 -ne (Get-FileSha256 $TestData)) {
        throw "Frozen MATH-500 test hash differs from its split manifest."
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
        throw "Frozen MATH-500 split audit is stale or not a zero-overlap PASS."
    }
}

function Require-No-PilotArtifacts {
    param([string[]]$Paths)
    $Existing = @($Paths | Where-Object { Test-Path $_ })
    if ($Existing.Count -gt 0) {
        throw (
            "MATH-500 scorer-pilot artifacts already exist: " +
            ($Existing -join ", ") +
            ". Inspect/archive them before a clean rerun; this wrapper will not overwrite evidence."
        )
    }
}

New-Item -ItemType Directory -Force -Path results | Out-Null
New-Item -ItemType Directory -Force -Path runs | Out-Null

$State = Read-FrozenState
$Ledger = $State.Ledger
$Head = [string]$State.Head

Require-LLaDAParity $Ledger $Head

$ModelEntry = @($Ledger.models | Where-Object { $_.adapter -eq "llada" })
$DatasetEntry = @($Ledger.datasets | Where-Object { $_.name -eq "math500" })
if ($ModelEntry.Count -ne 1) { throw "Revision ledger must contain exactly one LLaDA entry." }
if ($DatasetEntry.Count -ne 1) { throw "Revision ledger must contain exactly one MATH-500 entry." }

$ModelId = [string]$ModelEntry[0].model_id
$ModelSha = [string]$ModelEntry[0].resolved_sha
$DatasetSha = [string]$DatasetEntry[0].resolved_sha
if (-not $ModelSha -or $ModelSha -eq "null") { throw "Ledger has no frozen LLaDA SHA." }
if (-not $DatasetSha -or $DatasetSha -eq "null") { throw "Ledger has no frozen MATH-500 SHA." }

$CalData = "data/processed/math500/calibration.jsonl"
$TestData = "data/processed/math500/test.jsonl"
$SplitManifest = "data/processed/math500/split_manifest.json"
$SplitAudit = "data/processed/math500/split_audit.json"
$Raw = "runs/llada_math500_scorer_pilot.raw.jsonl"
$Scored = "runs/llada_math500_scorer_pilot.scored.jsonl"
$Review = "results/math500_scorer_review.md"

if ($Stage -in @("prepare", "all")) {
    Require-ScorerAudit $Head

    $SplitArtifacts = @($CalData, $TestData, $SplitManifest, $SplitAudit)
    $ExistingSplitArtifacts = @($SplitArtifacts | Where-Object { Test-Path $_ })
    if ($ExistingSplitArtifacts.Count -eq 0) {
        Run-Step "Prepare frozen MATH-500 calibration/test pair" {
            & $Python scripts/prepare_eval_pair.py --dataset math500 --revision $DatasetSha
        }
    }
    elseif ($ExistingSplitArtifacts.Count -ne $SplitArtifacts.Count) {
        throw (
            "Partial MATH-500 split artifacts exist: " +
            ($ExistingSplitArtifacts -join ", ") +
            ". Inspect them before scorer gating; this wrapper will not overwrite frozen data."
        )
    }

    Assert-Math500FrozenPair $CalData $TestData $SplitManifest $SplitAudit $DatasetSha

    Run-Step "MATH-500 LLaDA CUDA preflight" {
        & $Python scripts/preflight.py --model-config results/frozen_models/llada.yaml --data $CalData --require-cuda
    }
}

if ($Stage -in @("pilot", "all")) {
    Require-ScorerAudit $Head
    Assert-Math500FrozenPair $CalData $TestData $SplitManifest $SplitAudit $DatasetSha

    Require-No-PilotArtifacts @(
        $Raw,
        "$Raw.audit.json",
        $Scored,
        "$Scored.audit.json",
        $Review,
        "results/math500_scorer_manual_review.json"
    )

    Run-Step "25-example LLaDA MATH-500 calibration pilot" {
        & $Python scripts/run_generation.py --adapter llada --model $ModelId --revision $ModelSha --input $CalData --output $Raw --limit 25 --seed 2027 --steps 128 --max-new-tokens 128 --block-length 128 --temperature 0 --llada-cfg-scale 0 --llada-remasking low_confidence
    }
    Run-Step "Audit raw MATH-500 scorer pilot" {
        & $Python scripts/audit_run.py --run $Raw --stage raw --require-revision
    }
    Run-Step "Score MATH-500 calibration pilot" {
        & $Python scripts/score_outputs.py --run $Raw --input $CalData --output $Scored
    }
    Run-Step "Audit scored MATH-500 scorer pilot" {
        & $Python scripts/audit_run.py --run $Scored --stage scored --require-revision
    }
    Run-Step "Export deterministic 25-example MATH-500 review sheet" {
        & $Python scripts/export_math500_scorer_review.py --scored $Scored --input $CalData --count 25 --seed 2027 --output $Review
    }

    Write-Host ""
    Write-Host "AUTOMATED MATH-500 SCORER PILOT COMPLETE." -ForegroundColor Green
    Write-Host "Manual gate remains. Review: $Review"
    Write-Host "Then record the human attestation with scripts/record_math500_scorer_review.py."
}

Write-Host ""
Write-Host "Frozen commit: $Head"
Write-Host "Automatic scorer audit: results/math500_scorer_audit.json"
Write-Host "Calibration review sheet: $Review (after pilot)"
