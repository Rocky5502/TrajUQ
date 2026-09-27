param(
    [ValidateSet("sdar","dream","all")]
    [string]$Model = "all",
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
    if (-not (Test-Path $Artifact) -or -not (Test-Path $AuditPath)) { return $false }
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

function Read-FrozenState {
    $LedgerPath = "results/revision_ledger.json"
    $CpuPath = "results/cpu_verification.json"
    $GatedPath = "results/gated_dataset_access.json"
    if (-not (Test-Path $LedgerPath)) {
        throw "Missing $LedgerPath. Run scripts/freeze_preprimary_windows.ps1 first."
    }
    if (-not (Test-Path $CpuPath)) {
        throw "Missing $CpuPath. Run scripts/freeze_preprimary_windows.ps1 first."
    }
    if (-not (Test-Path $GatedPath)) {
        throw "Missing $GatedPath. Run scripts/freeze_preprimary_windows.ps1 first."
    }

    $Ledger = Get-Content $LedgerPath -Raw | ConvertFrom-Json
    $Cpu = Get-Content $CpuPath -Raw | ConvertFrom-Json
    $Gated = Get-Content $GatedPath -Raw | ConvertFrom-Json
    $Head = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $Head) {
        throw "Unable to resolve current git commit."
    }
    $Dirty = (& git status --porcelain) -join "`n"
    if ($LASTEXITCODE -ne 0) { throw "Unable to inspect git worktree." }
    if ($Dirty) {
        throw "Worktree is dirty. Restore the frozen clean checkout before parity."
    }

    if ($Ledger.protocol -ne "trajuq_revision_ledger_v2" -or $Ledger.status -ne "PASS") {
        throw "Revision ledger is not a protocol-v2 PASS."
    }
    if ($Cpu.protocol -ne "trajuq_cpu_verification_v1" -or $Cpu.status -ne "PASS" -or $Cpu.git_clean -ne $true) {
        throw "CPU verification is not a clean protocol-v1 PASS."
    }
    if ($Ledger.git_commit -ne $Head -or $Cpu.git_commit -ne $Head) {
        throw "Frozen CPU/revision artifacts do not match current HEAD $Head."
    }
    $LedgerHash = (Get-FileHash $LedgerPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if (
        $Gated.protocol -ne "trajuq_gated_dataset_access_v1" -or
        $Gated.status -ne "PASS" -or
        $Gated.git_clean -ne $true -or
        $Gated.git_commit -ne $Head -or
        $Gated.revision_ledger_sha256 -ne $LedgerHash
    ) {
        throw "Gated dataset access is not a clean revision-bound PASS at current HEAD."
    }

    return @{
        Ledger = $Ledger
        Head = $Head
    }
}

function Require-ExistingParity {
    param(
        [string]$Path,
        [string]$Adapter,
        [string]$Revision,
        [string]$Head
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
    $InstrumentedAuditOk = Test-RunAuditPass $Instrumented "raw" $Head
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
        $Parity.run_provenance.adapter -eq $Adapter -and
        $Parity.run_provenance.model_revision -eq $Revision -and
        $ReferenceHashOk -and
        $InstrumentedHashOk -and
        $InstrumentedAuditOk
    )
    if (-not $Pass) {
        throw "Existing $Path is not a clean hash/audit-bound frozen parity PASS. Inspect it before rerunning."
    }
    return $true
}

function Require-No-PartialArtifacts {
    param([string[]]$Paths)
    $Existing = @($Paths | Where-Object { Test-Path $_ })
    if ($Existing.Count -gt 0) {
        throw (
            "Partial parity artifacts already exist: " +
            ($Existing -join ", ") +
            ". Inspect/archive them before a clean rerun; this wrapper will not overwrite evidence."
        )
    }
}

function Run-SDARParity {
    param($Ledger, [string]$Head, [string]$CalData)

    $Entry = @($Ledger.models | Where-Object { $_.adapter -eq "sdar" })
    if ($Entry.Count -ne 1) { throw "Revision ledger must contain exactly one SDAR entry." }
    $ModelId = [string]$Entry[0].model_id
    $Revision = [string]$Entry[0].resolved_sha
    if (-not $Revision -or $Revision -eq "null") { throw "Ledger has no frozen SDAR SHA." }

    $Reference = "runs/sdar_parity_reference.jsonl"
    $Instrumented = "runs/sdar_parity_trajuq.raw.jsonl"
    $ParityPath = "results/sdar_parity.json"
    $Overhead = "results/sdar_instrumentation_overhead.json"

    if (Require-ExistingParity $ParityPath "sdar" $Revision $Head) {
        Write-Host "SDAR parity already PASS at frozen HEAD; preserving existing evidence." -ForegroundColor Green
        if (-not (Test-Path $Overhead) -and (Test-Path $Reference) -and (Test-Path $Instrumented)) {
            Run-Step "SDAR instrumentation overhead" {
                & $Python scripts/compare_instrumentation_overhead.py --reference $Reference --instrumented $Instrumented --output $Overhead
            }
        }
        return
    }

    Require-No-PartialArtifacts @($Reference, $Instrumented, "$Instrumented.audit.json", $Overhead)

    Run-Step "SDAR 20-prompt native reference" {
        & $Python scripts/run_sdar_reference.py --model $ModelId --revision $Revision --input $CalData --output $Reference --limit 20 --seed 2027 --max-new-tokens 128 --block-length 4 --denoising-steps 4 --temperature 1.0 --top-k 0 --top-p 1.0 --remasking-strategy low_confidence_dynamic --confidence-threshold 0.85
    }
    Run-Step "SDAR 20-prompt TrajUQ telemetry" {
        & $Python scripts/run_generation.py --adapter sdar --model $ModelId --revision $Revision --input $CalData --output $Instrumented --limit 20 --seed 2027 --max-new-tokens 128 --block-length 4 --sdar-denoising-steps 4 --temperature 1.0 --sdar-top-k 0 --top-p 1.0 --sdar-remasking-strategy low_confidence_dynamic --sdar-confidence-threshold 0.85
    }
    Run-Step "SDAR raw parity-run audit" {
        & $Python scripts/audit_run.py --run $Instrumented --stage raw --require-revision
    }
    Run-Step "SDAR exact token parity v3" {
        & $Python scripts/compare_run_parity.py --reference $Reference --instrumented $Instrumented --require-token-ids --require-provenance --output $ParityPath
    }
    if (-not (Require-ExistingParity $ParityPath "sdar" $Revision $Head)) {
        throw "SDAR parity did not produce a reusable PASS artifact."
    }
    Run-Step "SDAR instrumentation overhead" {
        & $Python scripts/compare_instrumentation_overhead.py --reference $Reference --instrumented $Instrumented --output $Overhead
    }
}

function Run-DreamParity {
    param($Ledger, [string]$Head, [string]$CalData)

    $Entry = @($Ledger.models | Where-Object { $_.adapter -eq "dream" })
    if ($Entry.Count -ne 1) { throw "Revision ledger must contain exactly one Dream entry." }
    $ModelId = [string]$Entry[0].model_id
    $Revision = [string]$Entry[0].resolved_sha
    if (-not $Revision -or $Revision -eq "null") { throw "Ledger has no frozen Dream SHA." }

    $Reference = "runs/dream_parity_reference.jsonl"
    $Instrumented = "runs/dream_parity_trajuq.raw.jsonl"
    $ParityPath = "results/dream_parity.json"
    $Overhead = "results/dream_instrumentation_overhead.json"

    if (Require-ExistingParity $ParityPath "dream" $Revision $Head) {
        Write-Host "Dream parity already PASS at frozen HEAD; preserving existing evidence." -ForegroundColor Green
        if (-not (Test-Path $Overhead) -and (Test-Path $Reference) -and (Test-Path $Instrumented)) {
            Run-Step "Dream instrumentation overhead" {
                & $Python scripts/compare_instrumentation_overhead.py --reference $Reference --instrumented $Instrumented --output $Overhead
            }
        }
        return
    }

    Require-No-PartialArtifacts @($Reference, $Instrumented, "$Instrumented.audit.json", $Overhead)

    Run-Step "Dream 20-prompt native reference" {
        & $Python scripts/run_dream_reference.py --model $ModelId --revision $Revision --input $CalData --output $Reference --limit 20 --seed 2027 --steps 128 --max-new-tokens 128 --temperature 0 --top-p 1.0 --alg entropy --alg-temp 0.1
    }
    Run-Step "Dream 20-prompt TrajUQ telemetry" {
        & $Python scripts/run_generation.py --adapter dream --model $ModelId --revision $Revision --input $CalData --output $Instrumented --limit 20 --seed 2027 --steps 128 --max-new-tokens 128 --temperature 0 --top-p 1.0 --dream-alg entropy --dream-alg-temp 0.1
    }
    Run-Step "Dream raw parity-run audit" {
        & $Python scripts/audit_run.py --run $Instrumented --stage raw --require-revision
    }
    Run-Step "Dream exact token parity v3" {
        & $Python scripts/compare_run_parity.py --reference $Reference --instrumented $Instrumented --require-token-ids --require-provenance --output $ParityPath
    }
    if (-not (Require-ExistingParity $ParityPath "dream" $Revision $Head)) {
        throw "Dream parity did not produce a reusable PASS artifact."
    }
    Run-Step "Dream instrumentation overhead" {
        & $Python scripts/compare_instrumentation_overhead.py --reference $Reference --instrumented $Instrumented --output $Overhead
    }
}

New-Item -ItemType Directory -Force -Path results | Out-Null
New-Item -ItemType Directory -Force -Path runs | Out-Null

$State = Read-FrozenState
$Ledger = $State.Ledger
$Head = [string]$State.Head

$DatasetEntry = @($Ledger.datasets | Where-Object { $_.name -eq "gsm8k" })
if ($DatasetEntry.Count -ne 1) { throw "Revision ledger must contain exactly one GSM8K entry." }
$DatasetSha = [string]$DatasetEntry[0].resolved_sha
if (-not $DatasetSha -or $DatasetSha -eq "null") { throw "Ledger has no frozen GSM8K SHA." }

$CalData = "data/processed/gsm8k/calibration.jsonl"
$TestData = "data/processed/gsm8k/test.jsonl"
$SplitManifest = "data/processed/gsm8k/split_manifest.json"
$SplitAudit = "data/processed/gsm8k/split_audit.json"
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
        ". Inspect them before parity; this wrapper will not overwrite frozen data."
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
        $Audit.protocol -ne "trajuq_split_pair_audit_v1" -or
        $Audit.status -ne "PASS" -or
        $Audit.requested_revision -ne $DatasetSha -or
        $Audit.calibration_sha256 -ne (Get-FileSha256 $CalData) -or
        $Audit.test_sha256 -ne (Get-FileSha256 $TestData) -or
        $Audit.id_overlap_count -ne 0 -or
        $Audit.exact_prompt_overlap_count -ne 0
    ) {
        throw "Existing GSM8K split audit is not a zero-overlap PASS."
    }
    Write-Host "Frozen GSM8K pair already present and hash-valid; preserving it." -ForegroundColor Green
}

if ($Model -in @("sdar", "all")) {
    Run-SDARParity $Ledger $Head $CalData
}
if ($Model -in @("dream", "all")) {
    Run-DreamParity $Ledger $Head $CalData
}

Write-Host ""
Write-Host "REQUESTED REMAINING PARITY GATES PASS" -ForegroundColor Green
Write-Host "Frozen commit: $Head"
Write-Host "SDAR artifact: results/sdar_parity.json (when requested)"
Write-Host "Dream artifact: results/dream_parity.json (when requested)"
Write-Host "Next: run python scripts/primary_readiness.py only after LLaDA pilot/manual review and MATH-500 gates are also complete."
