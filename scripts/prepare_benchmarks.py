#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def mc_prompt(question: str, choices: list[str], labels: list[str] | None = None) -> str:
    if labels is None:
        labels = [chr(ord("A") + i) for i in range(len(choices))]
    lines = [question.strip(), ""]
    lines += [f"{lab}. {text}" for lab, text in zip(labels, choices)]
    lines += ["", "Reason carefully and finish with: FINAL: <choice letter>"]
    return "\n".join(lines)


def normalize(name: str, row: dict, idx: int) -> dict:
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
        # Deterministic rotation: correct answer is not always option A.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    try:
        from datasets import load_from_disk
    except ImportError as e:
        raise SystemExit('Install extras first: pip install -e ".[models]"') from e

    ds = load_from_disk(str(Path(args.cache_dir) / args.dataset))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.dataset}.jsonl"

    count = min(len(ds), args.limit) if args.limit is not None else len(ds)
    with out.open("w", encoding="utf-8") as f:
        for idx in range(count):
            rec = normalize(args.dataset, dict(ds[idx]), idx)
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"[TrajUQ] wrote {count} normalized records to {out}")


if __name__ == "__main__":
    main()
