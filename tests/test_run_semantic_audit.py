import json
import subprocess
import sys
from pathlib import Path

from trajuq.io import sha256_text
from trajuq.schema import ExampleRecord, GenerationResult, StepSummary


def current_git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def make_record() -> dict:
    features = {
        "trajectory_steps": 2.0,
        "decision_entropy_mean": 0.3,
        "entropy_mean": 0.4,
        "entropy_p90": 0.5,
        "entropy_slope": -0.1,
        "margin_mean": 0.4,
        "margin_min": 0.2,
        "js_mean": 0.1,
        "js_p90": 0.2,
        "js_slope": 0.01,
        "late_js_mean": 0.15,
        "revision_rate": 0.2,
        "late_revision_rate": 0.1,
        "mask_auc": 0.5,
        "late_mask_fraction": 0.2,
        "commitment_time_mean": 0.4,
    }
    result = GenerationResult(
        text="FINAL: 1",
        token_ids=[1],
        features=features,
        steps=[
            StepSummary(
                step=0,
                entropy_mean=0.4,
                entropy_p90=0.5,
                confidence_mean=0.6,
                margin_mean=0.4,
                mask_fraction=1.0,
                revision_fraction=0.0,
                js_mean=0.0,
                active_fraction=1.0,
                active_count=1,
            )
        ],
        metadata={
            "model_revision": "sha",
            "trajectory_semantics": "active_unresolved_positions_v4",
        },
    )
    return ExampleRecord(
        example_id="x",
        dataset="synthetic",
        prompt_sha256=sha256_text("p"),
        model="m",
        adapter="llada",
        seed=2027,
        generation=result,
        metadata={
            "model_revision": "sha",
            "source_git_commit": current_git_head(),
            "source_git_clean": True,
        },
    ).to_dict()


def run_audit(tmp_path: Path, record: dict) -> subprocess.CompletedProcess:
    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            "scripts/audit_run.py",
            "--run",
            str(path),
            "--stage",
            "raw",
            "--require-revision",
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_run_audit_accepts_coherent_frozen_raw_artifact(tmp_path):
    result = run_audit(tmp_path, make_record())

    assert result.returncode == 0
    assert '"status": "PASS"' in result.stdout
    assert '"model_revisions": [' in result.stdout
    assert '"seeds": [' in result.stdout


def test_run_audit_rejects_other_source_commit(tmp_path):
    record = make_record()
    record["metadata"]["source_git_commit"] = "old-commit"
    result = run_audit(tmp_path, record)

    assert result.returncode != 0
    assert "source git commit differs" in result.stdout


def test_run_audit_rejects_bounded_feature_violation(tmp_path):
    record = make_record()
    record["generation"]["features"]["js_p90"] = 1.2
    result = run_audit(tmp_path, record)
    assert result.returncode != 0
    assert "outside [0,1]" in result.stdout


def test_run_audit_rejects_nested_outcome_key(tmp_path):
    record = make_record()
    record["generation"]["metadata"]["gold_answer"] = "secret"
    result = run_audit(tmp_path, record)
    assert result.returncode != 0
    assert "outcome-like metadata keys" in result.stdout
