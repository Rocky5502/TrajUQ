from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import finalize_rq4_evidence as finalizer


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_incomplete_generation_blocks_before_plan_or_outcomes(monkeypatch, tmp_path: Path):
    generation_root = tmp_path / "generation"
    generation_root.mkdir()
    addendum = generation_root / "results/rq4/sampling_freeze_addendum.json"
    qualification = generation_root / "results/rq4/sampling_qualification.json"
    write_json(addendum, {})
    write_json(qualification, {})
    monkeypatch.setattr(
        finalizer,
        "git",
        lambda root, *args: (
            finalizer.EXPECTED_GENERATION_COMMIT if args[-1] == "HEAD" else ""
        ),
    )
    monkeypatch.setattr(
        finalizer,
        "sha256_file",
        lambda path: (
            finalizer.EXPECTED_ADDENDUM_SHA256
            if Path(path) == addendum
            else finalizer.EXPECTED_QUALIFICATION_SHA256
            if Path(path) == qualification
            else hashlib.sha256(Path(path).read_bytes()).hexdigest()
        ),
    )
    manifest = tmp_path / "final_manifest.json"
    write_json(
        manifest,
        {
            "schema": "trajuq_tmlr_rq4_sampling_campaign_audit_v1",
            "status": "PASS",
            "complete": False,
            "rows": 19419,
            "expected_rows": 19420,
        },
    )
    with pytest.raises(ValueError, match="not exact-complete"):
        finalizer.require_generation_complete(generation_root, manifest)


def test_plan_rejects_generation_commands(monkeypatch, tmp_path: Path):
    script = tmp_path / "scripts" / "run_generation.py"
    script.parent.mkdir()
    script.write_text("print('forbidden')\n", encoding="utf-8")
    source = tmp_path / "source.json"
    write_json(source, {})
    generation_manifest = tmp_path / "generation.json"
    write_json(generation_manifest, {})
    stages = []
    for name in finalizer.EXPECTED_STAGE_ORDER:
        stages.append(
            {
                "name": name,
                "argv": ["python", "scripts/run_generation.py"],
                "script_sha256": finalizer.sha256_file(script),
                "outputs": [{"path": f"results/{name}.json"}],
            }
        )
    plan = tmp_path / "plan.json"
    write_json(
        plan,
        {
            "schema": "trajuq_rq4_downstream_plan_v1",
            "status": "PASS",
            "expected_generation_rows": 19420,
            "expected_generation_shards": 60,
            "generation_manifest_sha256": finalizer.sha256_file(generation_manifest),
            "sampling_addendum_sha256": finalizer.EXPECTED_ADDENDUM_SHA256,
            "sampling_qualification_sha256": finalizer.EXPECTED_QUALIFICATION_SHA256,
            "registered_science": {
                "bootstrap_replicates": 10000,
                "bootstrap_seed": 2027,
                "confidence_level": 0.95,
                "equal_cell_weight": True,
                "multiplicity_correction": "holm_bonferroni",
                "confirmatory_family": ["h4a", "h4b", "h4c"],
                "two_sided_alpha": 0.05,
            },
            "source_bindings": [{"path": "source.json", "sha256": finalizer.sha256_file(source)}],
            "stages": stages,
        },
    )
    with pytest.raises(ValueError, match="generation command forbidden"):
        finalizer.validate_plan(tmp_path, plan, generation_manifest)


def test_stage_order_and_registered_statistics_are_fixed():
    assert finalizer.EXPECTED_STAGE_ORDER == [
        "scoring",
        "feature_extraction",
        "calibration_fit",
        "heldout_evaluation",
        "paired_bootstrap",
        "multiplicity_correction",
        "hypothesis_gate",
        "macro_aggregate",
    ]
    text = Path("scripts/finalize_rq4_evidence.py").read_text(encoding="utf-8")
    assert "EXPECTED_ROWS = 19_420" in text
    assert "EXPECTED_SHARDS = 60" in text
    assert '"bootstrap_replicates": 10000' in text
    assert '"multiplicity_correction": "holm_bonferroni"' in text
    assert "refusing to overwrite immutable RQ4 final evidence manifest" in text
