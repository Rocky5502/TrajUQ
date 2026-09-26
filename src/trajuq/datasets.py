from __future__ import annotations

import hashlib
from typing import Any


NORMALIZER_VERSION = "trajuq-prompts-v1"


def mc_prompt(question: str, choices: list[str], labels: list[str] | None = None) -> str:
    if labels is None:
        labels = [chr(ord("A") + i) for i in range(len(choices))]
    lines = [question.strip(), ""]
    lines += [f"{lab}. {text}" for lab, text in zip(labels, choices)]
    lines += ["", "Reason carefully and finish with: FINAL: <choice letter>"]
    return "\n".join(lines)


def normalize_row(name: str, row: dict[str, Any], idx: int) -> dict[str, Any]:
    """Normalize one upstream benchmark row into TrajUQ's scorer-facing schema."""

    if name == "gsm8k":
        return {
            "id": str(idx),
            "dataset": name,
            "prompt": row["question"].strip()
            + "\n\nReason step by step and finish with: FINAL: <answer>",
            "reference": row["answer"],
            "metadata": {},
        }

    if name == "math500":
        return {
            "id": str(idx),
            "dataset": name,
            "prompt": row["problem"].strip()
            + "\n\nReason step by step and finish with: FINAL: <answer>",
            "reference": row["answer"],
            "metadata": {"subject": row.get("subject"), "level": row.get("level")},
        }

    if name == "arc_challenge":
        choices = row["choices"]
        return {
            "id": str(row.get("id", idx)),
            "dataset": name,
            "prompt": mc_prompt(row["question"], choices["text"], choices.get("label")),
            "reference": row["answerKey"],
            "metadata": {},
        }

    if name == "mmlu_pro":
        labels = [chr(ord("A") + i) for i in range(len(row["options"]))]
        if "answer_index" in row:
            ref_idx = int(row["answer_index"])
        else:
            raw = row["answer"]
            ref_idx = int(raw) if str(raw).isdigit() else labels.index(str(raw).strip())
        return {
            "id": str(row.get("question_id", idx)),
            "dataset": name,
            "prompt": mc_prompt(row["question"], row["options"], labels),
            "reference": labels[ref_idx],
            "metadata": {"category": row.get("category")},
        }

    if name == "truthfulqa":
        target = row.get("mc1_targets")
        if not target:
            raise ValueError("TruthfulQA row missing mc1_targets")
        choices = target["choices"]
        labels = [chr(ord("A") + i) for i in range(len(choices))]
        correct_idx = int(max(range(len(target["labels"])), key=lambda i: target["labels"][i]))
        return {
            "id": str(idx),
            "dataset": name,
            "prompt": mc_prompt(row["question"], choices, labels),
            "reference": labels[correct_idx],
            "metadata": {},
        }

    if name == "gpqa_diamond":
        choices = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        # Deterministic rotation avoids putting the correct option at A on every row.
        shift = idx % len(choices)
        rotated = choices[shift:] + choices[:shift]
        correct = chr(ord("A") + ((0 - shift) % len(choices)))
        return {
            "id": str(idx),
            "dataset": name,
            "prompt": mc_prompt(row["Question"], rotated),
            "reference": correct,
            "metadata": {"subdomain": row.get("Subdomain")},
        }

    raise ValueError(f"No normalizer for {name}")


def split_bucket(dataset: str, item_id: str, seed: str = "2027") -> float:
    digest = hashlib.sha256(f"{seed}:{dataset}:{item_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def namespace_record_id(dataset: str, source_split: str, item_id: str) -> str:
    """Create a globally unique evaluation ID without changing benchmark content.

    Official calibration and test splits often both number rows from zero. A bare
    row index therefore collides across splits and can make a clean evaluation
    look like label leakage. Namespacing IDs by dataset and immutable source
    split prevents that while preserving deterministic alignment.
    """
    dataset = str(dataset).strip()
    source_split = str(source_split).strip()
    item_id = str(item_id).strip()
    if not dataset or not source_split or not item_id:
        raise ValueError("dataset, source_split, and item_id must be non-empty")
    return f"{dataset}:{source_split}:{item_id}"
