from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_figure_spec_is_evidence_locked() -> None:
    spec = json.loads(
        (ROOT / "configs/figures/rq2_rq3_main_figures.json").read_text(encoding="utf-8")
    )
    assert len(spec["source_hashes"]) == 4
    assert spec["rq2"]["metrics"] == ["aurc", "auprc", "auroc", "ece15", "nll"]
    assert spec["rq3"]["primary_operating_point"] == "cap_050"
    assert spec["rq3"]["sensitivities"] == ["cap_025", "cap_075"]


def test_figure_renderer_has_fail_closed_and_claim_guards() -> None:
    text = (ROOT / "scripts/render_rq2_rq3_main_figures.py").read_text(
        encoding="utf-8"
    )
    ast.parse(text)
    assert "evidence drift" in text
    assert "no_fabricated_intervals" in text
    assert "negative_and_null_evidence_visible" in text
    assert "not an equivalence claim" in text
