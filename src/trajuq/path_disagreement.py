from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .scoring import extract_choice, extract_final_field, extract_number


DISAGREEMENT_SCHEMA_VERSION = "trajuq-path-disagreement-v1"
_MC_DATASETS = {"arc_challenge", "mmlu_pro", "truthfulqa", "gpqa_diamond"}


@dataclass(frozen=True)
class PathDisagreement:
    example_id: str
    dataset: str
    features: dict[str, float]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DISAGREEMENT_SCHEMA_VERSION,
            "example_id": self.example_id,
            "dataset": self.dataset,
            "features": self.features,
            "metadata": self.metadata,
        }


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _candidate_answer(dataset: str, text: str) -> str | None:
    name = dataset.lower()
    if name in _MC_DATASETS:
        return extract_choice(text)
    if name == "gsm8k":
        value = extract_number(text)
        return None if value is None else f"{value:.12g}"

    final = extract_final_field(text)
    target = final if final is not None else text
    target = _normalize_text(target)
    if not target:
        return None
    # Keep answer extraction conservative for open-ended/symbolic tasks. We do
    # not attempt mathematical equivalence here.
    return re.sub(r"\s+", " ", target).strip()


def _levenshtein(a: list[int], b: list[int]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current = [i]
        for j, token_b in enumerate(b, start=1):
            substitution = previous[j - 1] + (token_a != token_b)
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    substitution,
                )
            )
        previous = current
    return previous[-1]


def _common_prefix(a: list[int], b: list[int]) -> int:
    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += 1
    return count


def compare_paths(
    example_id: str,
    dataset: str,
    ar_text: str,
    diffusion_text: str,
    ar_token_ids: list[int] | None = None,
    diffusion_token_ids: list[int] | None = None,
) -> PathDisagreement:
    ar_norm = _normalize_text(ar_text)
    diff_norm = _normalize_text(diffusion_text)
    ar_answer = _candidate_answer(dataset, ar_text)
    diff_answer = _candidate_answer(dataset, diffusion_text)

    features: dict[str, float] = {
        "exact_text_disagreement": float(ar_norm != diff_norm),
        "parsed_answer_available": float(ar_answer is not None and diff_answer is not None),
        "parsed_answer_disagreement": float(
            ar_answer is not None
            and diff_answer is not None
            and ar_answer != diff_answer
        ),
    }

    token_metrics_available = ar_token_ids is not None and diffusion_token_ids is not None
    if token_metrics_available:
        a = [int(x) for x in ar_token_ids or []]
        b = [int(x) for x in diffusion_token_ids or []]
        denom = max(len(a), len(b), 1)
        distance = _levenshtein(a, b)
        prefix = _common_prefix(a, b)
        features.update(
            {
                "token_metrics_available": 1.0,
                "normalized_token_edit_distance": float(distance / denom),
                "token_common_prefix_fraction": float(prefix / denom),
                "normalized_length_difference": float(abs(len(a) - len(b)) / denom),
            }
        )
    else:
        features.update(
            {
                "token_metrics_available": 0.0,
                "normalized_token_edit_distance": 0.0,
                "token_common_prefix_fraction": 0.0,
                "normalized_length_difference": 0.0,
            }
        )

    metadata = {
        "ar_candidate_answer": ar_answer,
        "diffusion_candidate_answer": diff_answer,
        "answer_semantics": (
            "surface/parsed-answer disagreement only; AR is not ground truth and "
            "symbolic equivalence is not inferred"
        ),
    }
    return PathDisagreement(
        example_id=str(example_id),
        dataset=str(dataset),
        features=features,
        metadata=metadata,
    )
