import json
from pathlib import Path

import pytest

from trajuq.adapters.llama_ar import TELEMETRY_SCHEMA_VERSION, summarize_ar_trace


def test_rq4_ar_schema_is_outcome_blind_and_matches_adapter():
    schema = json.loads(
        Path("configs/schemas/rq4_ar_telemetry_v1.json").read_text(encoding="utf-8")
    )
    assert schema["schema"] == TELEMETRY_SCHEMA_VERSION
    assert schema["observation"] == "processed_prediction_scores_before_token_commit"
    assert schema["top_k"] == 20
    assert set(schema["outcome_fields_forbidden"]) >= {"correct", "reference", "target"}


def test_ar_summary_features_are_deterministic_and_prospectively_named():
    result = summarize_ar_trace(
        entropy=[0.4, 0.3, 0.2, 0.1],
        confidence=[0.5, 0.6, 0.7, 0.8],
        margin=[0.2, 0.3, 0.4, 0.5],
        selected_nll=[1.0, 0.8, 0.6, 0.4],
    )
    assert result["trajectory_steps"] == 4.0
    assert result["mean_token_entropy"] == pytest.approx(0.25)
    assert result["late_token_entropy"] == pytest.approx(0.1)
    assert result["mean_selected_token_nll"] == pytest.approx(0.7)
    assert result["min_token_margin"] == pytest.approx(0.2)
    assert result["entropy_slope"] < 0.0
    assert result["entropy_volatility"] > 0.0


def test_llama_ar_generation_contract_preserves_greedy_and_supports_frozen_sampling():
    text = Path("src/trajuq/adapters/llama_ar.py").read_text(encoding="utf-8")
    runner = Path("scripts/run_generation.py").read_text(encoding="utf-8")
    assert '"do_sample": bool(do_sample)' in text
    assert '"temperature": float(temperature)' in text
    assert '"top_p": float(top_p)' in text
    assert '"top_k": int(top_k)' in text
    assert '"output_scores": True' in text
    assert "def native_generate_token_ids(" in text
    assert "processed_prediction_scores_before_token_commit" in text
    assert 'parser.add_argument("--do-sample", action="store_true")' in runner
    assert 'do_sample = bool(getattr(args, "do_sample", False))' in runner
    assert '"correct"' not in text
    assert '"reference"' not in text
