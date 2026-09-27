from pathlib import Path

from trajuq.datasets import NORMALIZER_VERSION


def test_prompt_normalizer_version_is_frozen():
    assert NORMALIZER_VERSION == "trajuq-prompts-v2"


def test_prepare_eval_pair_requires_explicit_overwrite_flag():
    text = Path("scripts/prepare_eval_pair.py").read_text(encoding="utf-8")
    assert "--overwrite" in text
    assert "Refusing to replace frozen split artifacts" in text
    assert "--allow-unpinned-revision" in text


def test_prepare_eval_pair_requires_clean_source_git_provenance():
    text = Path("scripts/prepare_eval_pair.py").read_text(encoding="utf-8")
    assert "Unable to resolve source git commit" in text
    assert "Refusing to prepare frozen splits from a dirty git checkout" in text
    assert '"source_git_commit": source_git_commit' in text
    assert '"source_git_clean": source_git_clean' in text
    assert '"calibration_fraction": spec.get("calibration_fraction")' in text
    assert '"split_seed": spec.get("split_seed")' in text
