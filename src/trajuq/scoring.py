from __future__ import annotations

import math
import re
from functools import lru_cache


_FINAL_RE = re.compile(r"FINAL\s*:\s*(.+)", flags=re.IGNORECASE)
_NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)")


def extract_final_field(text: str) -> str | None:
    matches = _FINAL_RE.findall(text)
    if matches:
        return matches[-1].strip().splitlines()[0].strip()
    return None


def extract_choice(text: str) -> str | None:
    final = extract_final_field(text)
    target = final if final is not None else text
    # Prefer an isolated answer letter near the end.
    matches = re.findall(r"\b([A-J])\b", target.upper())
    return matches[-1] if matches else None


def _normalize_number(raw: str) -> float | None:
    raw = raw.strip().replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def extract_number(text: str) -> float | None:
    final = extract_final_field(text)
    target = final if final is not None else text
    matches = _NUMBER_RE.findall(target)
    return _normalize_number(matches[-1]) if matches else None


def gsm8k_reference_number(reference: str) -> float | None:
    tail = reference.split("####")[-1]
    return extract_number(tail)



@lru_cache(maxsize=1)
def _math500_metric():
    """Build the frozen judge-free MATH-500 equivalence metric lazily."""
    try:
        from math_verify.metric import math_metric
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    except ImportError as exc:
        raise RuntimeError(
            'MATH-500 scoring requires the pinned optional dependency: '
            'pip install -e ".[math]"'
        ) from exc

    return math_metric(
        gold_extraction_target=(
            LatexExtractionConfig(boxed_match_priority=0),
            ExprExtractionConfig(),
        ),
        pred_extraction_target=(
            LatexExtractionConfig(boxed_match_priority=0),
            ExprExtractionConfig(),
        ),
        aggregation_function=max,
        precision=6,
    )


def _boxed_math(text: str) -> str:
    value = str(text).strip()
    if "\\boxed{" in value:
        return value
    return "\\boxed{" + value + "}"


def score_math500(output_text: str, reference: str) -> bool:
    """Judge-free symbolic equivalence for the frozen MATH-500 protocol.

    The dataset reference and the model's explicit FINAL field are wrapped in a
    LaTeX boxed environment before parsing. If the model omitted FINAL, the full
    output is passed to Math-Verify's normal extraction logic.
    """

    metric = _math500_metric()
    gold = _boxed_math(reference)
    final = extract_final_field(output_text)
    prediction = _boxed_math(final) if final is not None else output_text
    try:
        grade, _ = metric([gold], [prediction])
    except Exception:
        return False
    return bool(grade == 1 or grade == 1.0)

def score_record(dataset: str, output_text: str, reference: str) -> tuple[bool, str]:
    name = dataset.lower()
    if name in {"arc_challenge", "mmlu_pro", "truthfulqa", "gpqa_diamond"}:
        pred = extract_choice(output_text)
        gold = str(reference).strip().upper()
        return (pred == gold, "choice_exact")

    if name == "gsm8k":
        pred = extract_number(output_text)
        gold = gsm8k_reference_number(reference)
        if pred is None or gold is None:
            return (False, "gsm8k_numeric_exact")
        return (abs(pred - gold) <= 1e-9, "gsm8k_numeric_exact")

    if name == "synthetic":
        pred_num = extract_number(output_text)
        gold_num = extract_number(reference)
        if pred_num is not None and gold_num is not None:
            return (abs(pred_num - gold_num) <= 1e-9, "synthetic_numeric_exact")
        return (output_text.strip() == reference.strip(), "synthetic_exact")

    if name == "math500":
        return (score_math500(output_text, reference), "math_verify_0.9.0_precision6")

    raise ValueError(f"No frozen scorer for dataset: {dataset}")
