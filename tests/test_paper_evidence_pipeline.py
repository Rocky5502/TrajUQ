from __future__ import annotations

import hashlib
import json
from pathlib import Path

import scripts.paper_evidence_pipeline as pipeline


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_progress_is_isolated_and_contains_no_scientific_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "repo"
    schema = root / "configs/schemas/progress_status_v1.json"
    write_json(schema, {"schema": "fixture"})
    source = root / "runs/rq2_progress.json"
    write_json(
        source,
        {
            "schema": "trajuq_rq2_progress_input_v1",
            "observed_at_utc": "2026-10-20T00:00:00Z",
            "state": "RUNNING",
            "completed_jobs": 80,
            "required_jobs": 120,
            "completed_seeds": [2027, 2028],
            "audit_pass_count": 80,
            "throughput_jobs_per_hour": 2.0,
            "eta_hours": 20.0,
        },
    )
    monkeypatch.setattr(pipeline, "ROOT", root)
    monkeypatch.setattr(pipeline, "PROGRESS_SCHEMA", schema)
    output_json = root / "results/generated/progress_status.json"
    output_tex = root / "paper/generated/progress_status.tex"
    result = pipeline.build_progress(source, output_json, output_tex, "paper-evidence-v1")

    assert result["science_authorization"] == "NOT_AUTHORIZED_FOR_SCIENTIFIC_CLAIMS"
    assert result["completion"]["completion_percent"] == 66.666667
    tex = output_tex.read_text(encoding="utf-8")
    assert "NOT AUTHORIZED FOR SCIENTIFIC CLAIMS" in tex
    assert "AUROC" not in tex


def test_progress_rejects_scientific_result_fields(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    source = root / "runs/rq2_progress.json"
    write_json(
        source,
        {
            "schema": "trajuq_rq2_progress_input_v1",
            "observed_at_utc": "2026-10-20T00:00:00Z",
            "state": "RUNNING",
            "completed_jobs": 80,
            "required_jobs": 120,
            "auroc": 0.9,
        },
    )
    monkeypatch.setattr(pipeline, "ROOT", root)
    try:
        pipeline.progress_input(source)
    except pipeline.EvidenceError as exc:
        assert "non-operational fields" in str(exc)
    else:
        raise AssertionError("scientific progress field was accepted")


def test_rq2_final_manifest_requires_all_120_seed_jobs(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    script = root / "scripts/analyze.py"
    artifact = root / "results/source.json"
    output = root / "paper/generated/rq2.tex"
    for path, data in ((script, b"pass\n"), (artifact, b"{}\n"), (output, b"% generated\n")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = root / "results/generated/final/rq2_manifest.json"
    gates = {
        "generation_complete": True,
        "audits_pass": True,
        "uniqueness_pass": True,
        "completeness_pass": True,
        "scoring_complete": True,
        "analysis_complete": True,
        "claim_gate_decided": True,
        "registry_frozen": True,
        "seed_jobs_expected": 120,
        "seed_jobs_complete": 119,
    }
    write_json(
        manifest,
        {
            "schema": "trajuq_final_rq_evidence_v1",
            "registry_version": "paper-evidence-v1",
            "rq": "RQ2",
            "status": "FINAL",
            "experiment_commit": "a" * 40,
            "generated_at_utc": "2026-10-20T00:00:00Z",
            "analysis_script": "scripts/analyze.py",
            "analysis_script_sha256": sha256(script),
            "audit_status": "PASS",
            "identifiers": {"models": ["llada"], "datasets": ["gsm8k"], "seeds": [2027]},
            "gates": gates,
            "sources": [{"path": "results/source.json", "sha256": sha256(artifact)}],
            "outputs": [{"path": "paper/generated/rq2.tex", "sha256": sha256(output)}],
        },
    )
    monkeypatch.setattr(pipeline, "ROOT", root)
    failures = pipeline.validate_final_manifest(manifest, "RQ2", "paper-evidence-v1")
    assert "RQ2 gates.seed_jobs_complete must equal 120" in failures

    gates["seed_jobs_complete"] = 120
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["gates"] = gates
    write_json(manifest, payload)
    assert pipeline.validate_final_manifest(manifest, "RQ2", "paper-evidence-v1") == []
