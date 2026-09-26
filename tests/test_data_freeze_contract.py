from pathlib import Path

from trajuq.datasets import NORMALIZER_VERSION


def test_prompt_normalizer_version_is_frozen():
    assert NORMALIZER_VERSION == "trajuq-prompts-v1"


def test_prepare_eval_pair_requires_explicit_overwrite_flag():
    text = Path("scripts/prepare_eval_pair.py").read_text(encoding="utf-8")
    assert "--overwrite" in text
    assert "Refusing to replace frozen split artifacts" in text
    assert "--allow-unpinned-revision" in text
