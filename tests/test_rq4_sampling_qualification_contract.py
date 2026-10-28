from __future__ import annotations

import ast
from pathlib import Path


def test_sampling_freeze_is_non_overwriting_and_binds_original_freeze() -> None:
    text = Path("scripts/freeze_rq4_sampling.py").read_text(encoding="utf-8")
    assert "Refusing to overwrite immutable RQ4 sampling freeze evidence" in text
    assert "Sampled RQ4 artifacts already exist" in text
    assert "EXPECTED_ORIGINAL_FREEZE_SHA256" in text
    assert '"scientific_settings_changed": False' in text
    assert '"sampled_outcomes_observed": False' in text
    assert 'manifest["expected_generations"] != 19420' in text
    ast.parse(text)


def test_sampling_qualification_is_exact_outcome_blind_parity() -> None:
    text = Path("scripts/qualify_rq4_sampling.py").read_text(encoding="utf-8")
    assert "Refusing to overwrite immutable RQ4 sampling qualification evidence" in text
    assert "native_generate_token_ids" in text
    assert '"exact_token_parity": not parity_failures' in text
    assert '"correct": None' in text
    assert '"reference": None' in text
    assert '"go_no_go": "GO" if status == "PASS" else "NO_GO"' in text
    ast.parse(text)


def test_sampling_provenance_distinguishes_registration_from_executor_support() -> None:
    text = Path("docs/RQ4_SAMPLING_PROVENANCE.md").read_text(encoding="utf-8")
    assert "scientific settings were prospectively registered" in text
    assert "original executor did not support" in text
    assert "Greedy RQ4 evidence existed before" in text
    assert "No sampled RQ4 outcome existed or was inspected before the addendum" in text
    assert "not represented as part of the original executable" in text
    assert "changed no seed, temperature, top-p, top-k, token limit" in text
    assert "12/12 rows" in text
    assert "exact native-versus-instrumented token-ID" in text
