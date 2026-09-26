#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from trajuq.io import read_jsonl
from trajuq.risk import raw_trajectory_risk


def fmt(value) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a stratified manual-review sheet for pilot trajectories."
    )
    parser.add_argument("--scored", required=True)
    parser.add_argument("--output", default="results/pilot_trace_review.md")
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()

    scored_path = Path(args.scored)
    rows = list(read_jsonl(scored_path))
    if not rows:
        raise SystemExit("No scored rows")

    ranked = []
    for row in rows:
        if row.get("correct") not in (True, False):
            raise SystemExit("All rows must be scored before trace-review export")
        ranked.append((float(raw_trajectory_risk(row["generation"]["features"])), row))
    ranked.sort(key=lambda x: x[0])

    k = min(args.count, len(ranked))
    indices = [0] if k == 1 else sorted(
        set(int(round(v)) for v in np.linspace(0, len(ranked) - 1, k))
    )
    selected = [ranked[i] for i in indices]

    lines = [
        "# TrajUQ pilot manual trace review",
        "",
        "Selected across the training-free trajectory-risk ranking.",
        "Review logging correctness; do not redesign frozen features because of these outcomes.",
        "",
        f"- scored artifact: {scored_path}",
        f"- scored SHA-256: {sha256_file(scored_path)}",
        f"- selected records: {len(selected)}",
        "",
        "| ID | Correct | Raw risk | Decision H | Entropy p90 | JS p90 | Late rev. | Commit | Steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for risk, row in selected:
        feats = row["generation"]["features"]
        lines.append(
            "| {id} | {correct} | {risk} | {decision} | {entropy} | {js} | {rev} | {commit} | {steps} |".format(
                id=row["example_id"],
                correct="1" if row["correct"] else "0",
                risk=fmt(risk),
                decision=fmt(feats.get("decision_entropy_mean")),
                entropy=fmt(feats.get("entropy_p90")),
                js=fmt(feats.get("js_p90")),
                rev=fmt(feats.get("late_revision_rate")),
                commit=fmt(feats.get("commitment_time_mean")),
                steps=len(row["generation"].get("steps", [])),
            )
        )

    lines += ["", "## Per-example review", ""]
    for risk, row in selected:
        feats = row["generation"]["features"]
        step_rows = row["generation"].get("steps", [])
        generated = str(row["generation"]["text"])[:600].replace("|", "/").replace("\n", " ")
        revision = (
            row.get("metadata", {}).get("model_revision")
            or row["generation"].get("metadata", {}).get("model_revision")
        )
        lines += [
            f"### {row['example_id']}",
            "",
            f"- Correct: {row['correct']}",
            f"- Raw risk: {risk:.6f}",
            f"- Generated text: {generated}",
            f"- Model revision: {revision}",
            "- Feature snapshot: "
            f"decision_entropy={fmt(feats.get('decision_entropy_mean'))}, "
            f"entropy_p90={fmt(feats.get('entropy_p90'))}, "
            f"js_p90={fmt(feats.get('js_p90'))}, "
            f"late_revision={fmt(feats.get('late_revision_rate'))}, "
            f"commitment={fmt(feats.get('commitment_time_mean'))}.",
            "",
            "First/last step summaries:",
            "",
        ]
        compact = step_rows[:3] + (step_rows[-3:] if len(step_rows) > 3 else [])
        seen = set()
        for step in compact:
            step_id = step.get("step")
            if step_id in seen:
                continue
            seen.add(step_id)
            lines.append(
                f"- step {step_id}: H={fmt(step.get('entropy_mean'))}, "
                f"JS={fmt(step.get('js_mean'))}, mask={fmt(step.get('mask_fraction'))}, "
                f"rev={fmt(step.get('revision_fraction'))}, active={step.get('active_count')}"
            )
        lines += [
            "",
            "- [ ] Generated answer/parser looks sensible",
            "- [ ] Step count/active positions look plausible",
            "- [ ] Entropy/JS/revisions are not obvious logging artifacts",
            "- [ ] No prompt tokens or future inactive blocks appear in telemetry",
            "- [ ] Commitment behavior is plausible",
            "",
        ]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[TrajUQ] wrote {len(selected)}-example review sheet -> {out}")


if __name__ == "__main__":
    main()
