import json
from pathlib import Path
from types import SimpleNamespace

from scripts.run_repeated_samples import (
    repeated_seed_contract,
    resumable_partial_artifact,
)


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        adapter="llada",
        model="example/llada",
        revision="model-sha",
        temperature=0.2,
        steps=128,
        max_new_tokens=128,
        block_length=128,
        llada_cfg_scale=0.0,
        llada_remasking="low_confidence",
        dream_alg="entropy",
        dream_alg_temp=0.1,
        top_p=1.0,
        sdar_denoising_steps=4,
        sdar_top_k=0,
        sdar_remasking_strategy="low_confidence_dynamic",
        sdar_confidence_threshold=0.85,
    )


def _row(example_id: str, seed: int = 2027) -> dict:
    return {
        "example_id": example_id,
        "adapter": "llada",
        "model": "example/llada",
        "seed": seed,
        "metadata": {
            "source_git_commit": "abc",
            "source_git_clean": True,
        },
        "generation": {
            "metadata": {
                "model_revision": "model-sha",
                "temperature": 0.2,
                "steps_requested": 128,
                "gen_length": 128,
                "block_length": 128,
                "cfg_scale": 0.0,
                "remasking": "low_confidence",
            }
        },
    }


def test_resumable_partial_requires_exact_frozen_prefix(tmp_path: Path):
    path = tmp_path / "seed-2027.raw.jsonl"
    path.write_text(
        json.dumps(_row("a")) + "\n" + json.dumps(_row("b")) + "\n",
        encoding="utf-8",
    )
    contract = repeated_seed_contract(_args(), 2027, "abc")

    assert resumable_partial_artifact(path, ["a", "b", "c"], contract)
    assert not resumable_partial_artifact(path, ["b", "a", "c"], contract)


def test_resumable_partial_rejects_audited_or_wrong_contract(tmp_path: Path):
    path = tmp_path / "seed-2027.raw.jsonl"
    path.write_text(json.dumps(_row("a")) + "\n", encoding="utf-8")
    contract = repeated_seed_contract(_args(), 2027, "abc")

    path.with_suffix(path.suffix + ".audit.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    assert not resumable_partial_artifact(path, ["a", "b"], contract)

    path.with_suffix(path.suffix + ".audit.json").unlink()
    wrong = repeated_seed_contract(_args(), 2028, "abc")
    assert not resumable_partial_artifact(path, ["a", "b"], wrong)


def test_resumable_partial_accepts_complete_unaudited_prefix(tmp_path: Path):
    path = tmp_path / "seed-2027.raw.jsonl"
    path.write_text(
        json.dumps(_row("a")) + "\n" + json.dumps(_row("b")) + "\n",
        encoding="utf-8",
    )
    contract = repeated_seed_contract(_args(), 2027, "abc")

    assert resumable_partial_artifact(path, ["a", "b"], contract)
