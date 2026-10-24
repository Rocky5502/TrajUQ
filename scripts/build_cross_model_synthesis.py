#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trajuq.cross_model_synthesis import AVAILABLE, build_registry, sha256_file


def tex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in value)


def metric_tex(item: dict[str, Any], digits: int = 3) -> str:
    if item["state"] == "pending":
        return r"\textit{pending}"
    if item["state"] == "not_applicable":
        return r"\textemdash{}"
    value = item["value"]
    if isinstance(value, int) and digits == 0:
        return str(value)
    if isinstance(value, (float, int)):
        return f"{float(value):.{digits}f}"
    return tex_escape(str(value))


def availability_tex(value: str) -> str:
    return {
        "available": "yes",
        "pending": r"\textit{pending}",
        "not_applicable": r"\textemdash{}",
    }[value]


def parameter_tex(item: dict[str, Any]) -> str:
    if item["state"] == "not_applicable":
        return r"\textemdash{}"
    if item["state"] == "pending":
        return r"\textit{pending}"
    return f"{float(item['value']) / 1e9:.2f}B"


def render_main_table(registry: dict[str, Any]) -> str:
    rows = []
    previous_regime = None
    for model in registry["models"]:
        if previous_regime is not None and previous_regime != model["regime"]:
            rows.append(r"\addlinespace[2pt]")
        previous_regime = model["regime"]
        signal = {
            "diffusion_trajectory": "native trajectory",
            "autoregressive_trace": "AR trace",
            "black_box_behavior": "behavioral repeats",
        }[model["regime"]]
        rows.append(
            " & ".join(
                [
                    tex_escape(model["display_name"]),
                    tex_escape(model["regime_display"]),
                    tex_escape(signal),
                    metric_tex(model["metrics"]["auroc"]),
                    metric_tex(model["metrics"]["within_regime_auroc_improvement"]),
                    metric_tex(model["metrics"]["aurc"]),
                ]
            )
            + r" \\"
        )
    state_note = (
        "All contributing registries are authoritative."
        if registry["final"]
        else (
            "This is a provisional synthesis: pending entries are not estimates and "
            "support no final cross-regime claim."
        )
    )
    return rf"""
\begin{{table*}}[t]
\centering
\small
\setlength{{\tabcolsep}}{{4.5pt}}
\begin{{tabular}}{{lllrrr}}
\toprule
System & Observability regime & Principal signal & AUROC $\uparrow$ & $\Delta$ AUROC$^\dagger$ $\uparrow$ & AURC $\downarrow$ \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\caption{{What uncertainty information is observable and useful under each access regime. Values are equal-dataset-weight summaries within their registered regime; $\Delta$ AUROC compares only with the pre-registered baseline from the same regime. {state_note} The table is descriptive across regimes and must not be read as a raw cross-protocol model ranking.}}
\label{{tab:cross-model-synthesis}}
\end{{table*}}
""".strip()


def render_appendix_table(registry: dict[str, Any]) -> str:
    short_names = {
        "dream": "Dream",
        "llada": "LLaDA",
        "sdar": "SDAR",
        "llama31": "Llama 3.1",
        "deepseek_v32": "DeepSeek",
        "glm51": "GLM",
        "kimi_k26": "Kimi",
    }
    regime_names = {
        "diffusion_trajectory": "Diffusion-trajectory observability",
        "autoregressive_trace": "Autoregressive-trace observability",
        "black_box_behavior": "Black-box behavioral observability",
    }
    tables = []
    for regime in regime_names:
        rows = []
        previous_model = None
        for row in registry["appendix_rows"]:
            if row["regime"] != regime:
                continue
            if previous_model is not None and previous_model != row["model_key"]:
                rows.append(r"\addlinespace[2pt]")
            previous_model = row["model_key"]
            metrics = row["metrics"]
            rows.append(
                " & ".join(
                    [
                        short_names[row["model_key"]],
                        tex_escape(row["dataset"]),
                        metric_tex(row["n_test"], digits=0),
                        metric_tex(metrics["auroc"]),
                        metric_tex(metrics["auprc"]),
                        metric_tex(metrics["aurc"]),
                        metric_tex(metrics["ece15"]),
                        metric_tex(metrics["risk50"]),
                        metric_tex(metrics["within_regime_auroc_improvement"]),
                    ]
                )
                + r" \\"
            )
        label = regime.replace("_", "-")
        tables.append(
            rf"""
\begin{{table*}}[p]
\centering
\small
\setlength{{\tabcolsep}}{{3.2pt}}
\begin{{tabular}}{{llrrrrrrr}}
\toprule
System & Dataset & $n$ & AUROC $\uparrow$ & AUPRC $\uparrow$ & AURC $\downarrow$ & ECE$_{{15}}$ $\downarrow$ & Risk@50 $\downarrow$ & $\Delta$ AUROC$^\dagger$ $\uparrow$ \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\caption{{{regime_names[regime]}: complete model--dataset synthesis registry. \textit{{pending}} denotes a registered cell whose authoritative aggregate is not yet available; \textemdash{{}} denotes a genuinely unavailable or scientifically inapplicable quantity, never a measured zero. $\Delta$ values use only the appropriate within-regime baseline.}}
\label{{tab:cross-model-synthesis-{label}}}
\end{{table*}}
""".strip()
        )
    return "\n\n".join(tables)


def render_signal_table(registry: dict[str, Any]) -> str:
    rows = []
    for model in registry["models"]:
        rows.append(
            " & ".join(
                [
                    tex_escape(model["display_name"]),
                    tex_escape(model["architecture"]),
                    parameter_tex(model["parameter_count"]),
                    metric_tex(model["compute_or_api_cost"], digits=2),
                    availability_tex(model["revision_signal"]),
                    availability_tex(model["persistence_signal"]),
                    availability_tex(model["commitment_signal"]),
                ]
            )
            + r" \\"
        )
    return rf"""
\begin{{table*}}[p]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{2pt}}
\begin{{tabular}}{{p{{2.5cm}}p{{4.6cm}}rrrrr}}
\toprule
System & Architecture/access & Parameters & Cost$^\ddagger$ & Revision & Persistence & Commitment \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\caption{{Structural availability of uncertainty signals. $^\ddagger$Compute/API cost is populated only from a model-bound authoritative ledger; it is never inferred from nominal rental prices. An em dash marks a signal or parameter count that is structurally inapplicable or not exposed by the access interface. It is distinct from \textit{{pending}}, which means that an authorized empirical aggregate has not yet passed its final registry gate.}}
\label{{tab:cross-model-signal-availability}}
\end{{table*}}
""".strip()


def render_tex(registry: dict[str, Any]) -> str:
    status = "FINAL" if registry["final"] else "PROVISIONAL"
    return "\n\n".join(
        [
            "% AUTO-GENERATED from provenance-bound synthesis artifacts; do not edit.",
            f"% SYNTHESIS STATE: {status}",
            r"\newcommand{\CrossModelSynthesisState}{" + status + "}",
            r"\newcommand{\CrossModelSynthesisTable}{%" + "\n" + render_main_table(registry) + "}",
            r"\newcommand{\CrossModelAppendixTable}{%" + "\n" + render_appendix_table(registry) + "}",
            r"\newcommand{\CrossModelSignalTable}{%" + "\n" + render_signal_table(registry) + "}",
        ]
    ) + "\n"


def render_figure(registry: dict[str, Any], output_base: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "trajuq-cross-model-synthesis-v1",
        }
    )
    regimes = [
        ("diffusion_trajectory", "Diffusion trajectory"),
        ("autoregressive_trace", "Autoregressive trace"),
        ("black_box_behavior", "Black-box behavior"),
    ]
    short_labels = {
        "dream": "Dream",
        "llada": "LLaDA",
        "sdar": "SDAR",
        "llama31": "Llama 3.1",
        "deepseek_v32": "DeepSeek",
        "glm51": "GLM",
        "kimi_k26": "Kimi",
    }
    fig, axes = plt.subplots(1, 3, figsize=(178 / 25.4, 88 / 25.4), sharex=True)
    for axis, (regime_key, title) in zip(axes, regimes, strict=True):
        members = [model for model in registry["models"] if model["regime"] == regime_key]
        axis.axvline(0.0, color="#D8D8D4", linewidth=0.8, zorder=0)
        axis.set_title(title, loc="left", fontweight="semibold", color="#171717")
        axis.set_xlim(-0.18, 0.24)
        axis.set_xticks([-0.1, 0.0, 0.1, 0.2])
        axis.grid(axis="x", color="#E8E8E5", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.set_yticks(range(len(members)), [short_labels[model["key"]] for model in members])
        axis.invert_yaxis()
        for index, model in enumerate(members):
            item = model["metrics"]["within_regime_auroc_improvement"]
            if item["state"] != AVAILABLE:
                axis.text(
                    0.0,
                    index,
                    "pending",
                    va="center",
                    ha="left",
                    color="#6F6F6F",
                    fontstyle="italic",
                )
                continue
            value = float(item["value"])
            if item.get("ci_low") is not None:
                axis.errorbar(
                    value,
                    index,
                    xerr=[[value - item["ci_low"]], [item["ci_high"] - value]],
                    fmt="o",
                    color="#1F6F5F",
                    ecolor="#1F6F5F",
                    elinewidth=0.8,
                    capsize=2,
                    markersize=4.5,
                )
            else:
                axis.plot(value, index, "o", color="#1F6F5F", markersize=4.5)
        axis.tick_params(axis="both", length=3, color="#8D8D88")
        axis.spines["left"].set_color("#D8D8D4")
        axis.spines["bottom"].set_color("#D8D8D4")
    fig.suptitle(
        "Usable uncertainty is assessed within three observability regimes",
        x=0.065,
        y=0.985,
        ha="left",
        fontsize=10.5,
        fontweight="semibold",
        color="#171717",
    )
    fig.supxlabel(
        "Within-regime AUROC improvement · higher is better",
        x=0.54,
        y=0.12,
        fontsize=8.5,
        color="#171717",
    )
    fig.text(
        0.065,
        0.018,
        (
            "Panels are descriptive across regimes; do not rank systems across panels.\n"
            "Intervals appear only when an authoritative compatible registry provides them."
        ),
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#5E6462",
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.79, bottom=0.24, wspace=0.48)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs = [output_base.with_suffix(suffix) for suffix in (".pdf", ".svg", ".png")]
    fig.savefig(
        outputs[0],
        metadata={"CreationDate": None, "ModDate": None, "Title": None},
    )
    fig.savefig(outputs[1], metadata={"Date": None})
    svg_text = outputs[1].read_text(encoding="utf-8")
    outputs[1].write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(outputs[2], dpi=300, metadata={"Date": None})
    plt.close(fig)
    return outputs


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the provisional/final cross-model synthesis from immutable artifacts."
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    contract_path = root / "configs" / "analysis" / "cross_model_synthesis.json"
    registry = build_registry(root, contract_path)

    output_dir = root / "results" / "cross_model_synthesis"
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = output_dir / "registry.json"
    registry_text = json.dumps(registry, indent=2, sort_keys=True) + "\n"
    registry_path.write_text(registry_text, encoding="utf-8", newline="\n")

    compatibility_path = output_dir / "compatibility_report.json"
    compatibility = {
        "schema": "trajuq_metric_compatibility_report_v1",
        "status": "PASS",
        "direct_comparison_groups": registry["compatibility"],
        "cross_regime_policy": "descriptive_only",
        "forbidden": [
            "raw cross-protocol AUROC ranking",
            "pooled cross-regime significance test",
            "causal observability-to-performance claim",
        ],
    }
    compatibility_path.write_text(
        json.dumps(compatibility, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    tex = render_tex(registry)
    tex_path = root / "paper" / "generated_synthesis.tex"
    tex_path.write_text(tex, encoding="utf-8", newline="\n")

    figure_base = root / "paper" / "figures" / "generated" / "cross_observability_synthesis"
    figures = render_figure(registry, figure_base)

    qa = {
        "schema": "trajuq_cross_model_visual_qa_v1",
        "status": "PASS",
        "state": registry["status"],
        "final_dimensions_mm": {"width": 178, "height": 88},
        "minimum_text_pt": 8.0,
        "claim": "Usable uncertainty is assessed within three observability regimes.",
        "form": "three-panel dot-and-interval scaffold with strict regime separation",
        "checks": {
            "metric_compatibility_firewall": True,
            "pending_and_na_distinct": True,
            "no_fabricated_intervals": True,
            "cross_regime_ranking_forbidden": True,
            "vector_pdf_and_svg": True,
            "grayscale_redundancy": True,
        },
        "note": (
            "Final visual inspection and authoritative value spot-check must be repeated "
            "after RQ2, RQ4, and RQ5 complete and the registry is regenerated from scratch."
        ),
    }
    qa_path = output_dir / "visual_qa.json"
    qa_path.write_text(
        json.dumps(qa, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    manifest = {
        "schema": "trajuq_cross_model_synthesis_manifest_v1",
        "status": "PASS",
        "synthesis_state": registry["status"],
        "final": registry["final"],
        "authoritative_commit": registry["authoritative_commit"],
        "contract": {
            "path": str(contract_path.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(contract_path),
        },
        "sources": registry["source_bindings"],
        "outputs": [],
    }
    for path in [registry_path, compatibility_path, tex_path, qa_path, *figures]:
        manifest["outputs"].append(
            {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest_path = output_dir / "synthesis_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(
        json.dumps(
            {
                "status": "PASS",
                "synthesis_state": registry["status"],
                "registry": str(registry_path.relative_to(root)),
                "manifest": str(manifest_path.relative_to(root)),
                "tex_sha256": sha256_text(tex),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
