from __future__ import annotations

import math
import os
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

    gold_targets = (
        LatexExtractionConfig(boxed_match_priority=0),
        ExprExtractionConfig(),
    )
    pred_targets = (
        LatexExtractionConfig(boxed_match_priority=0),
        ExprExtractionConfig(),
    )

    if os.name != "nt":
        return math_metric(
            gold_extraction_target=gold_targets,
            pred_extraction_target=pred_targets,
            aggregation_function=max,
            precision=6,
        )

    # math-verify 0.9.0's timeout decorator uses multiprocessing in a way
    # that is incompatible with Windows spawn semantics. Keep the exact same
    # parser/verifier semantics for the Windows CPU gate, but disable only the
    # library's internal timeout wrapper. Authoritative experiment scoring runs
    # on Linux and continues to use the official math_metric path above.
    from math_verify import parse, verify

    def windows_metric(
        golds: list[str],
        predictions: list[str],
    ) -> tuple[float, None]:
        extracted_golds = [
            parse(
                gold,
                extraction_config=gold_targets,
                parsing_timeout=None,
            )
            for gold in golds
        ]
        extracted_predictions = [
            parse(
                prediction,
                extraction_config=pred_targets,
                parsing_timeout=None,
            )
            for prediction in predictions
        ]
        if any(len(items) == 0 for items in extracted_golds):
            raise ValueError("Math-Verify could not parse one or more gold targets")

        grades: list[float] = []
        for prediction_items in extracted_predictions:
            matched = any(
                verify(
                    gold_items,
                    prediction_items,
                    float_rounding=6,
                    timeout_seconds=None,
                )
                for gold_items in extracted_golds
            )
            grades.append(1.0 if matched else 0.0)
        return (max(grades, default=0.0), None)

    return windows_metric


def _boxed_math(text: str) -> str:
    value = str(text).strip()
    if "\\boxed{" in value:
        return value
    return "\\boxed{" + value + "}"


def _math500_prediction_text(text: str) -> str:
    final = extract_final_field(text)
    return _boxed_math(final) if final is not None else str(text)


def math500_equivalent(left: str, right: str) -> bool:
    """Reference-free symbolic equivalence used for MATH-500 sample clustering.

    Exact normalized text equality is accepted first. Otherwise the same pinned
    Math-Verify metric used by the scorer is applied pairwise. No benchmark
    reference answer or correctness label is needed.
    """

    left_text = str(left).strip()
    right_text = str(right).strip()
    if not left_text or not right_text:
        return False
    if left_text == right_text:
        return True

    metric = _math500_metric()
    try:
        grade, _ = metric(
            [_math500_prediction_text(left_text)],
            [_math500_prediction_text(right_text)],
        )
    except Exception:
        return False
    return bool(grade == 1 or grade == 1.0)


def score_math500(output_text: str, reference: str) -> bool:
    """Judge-free symbolic equivalence for the frozen MATH-500 protocol.

    The benchmark reference is always treated as an answer expression and
    boxed before parsing. A model prediction uses its explicit FINAL field when
    available; otherwise Math-Verify sees the complete output so its normal
    extraction rules can operate.
    """

    metric = _math500_metric()
    gold = _boxed_math(reference)
    prediction = _math500_prediction_text(output_text)
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



def canonical_prediction(dataset: str, output_text: str) -> str | None:
    """Canonical prediction used only for label-free repeated-sampling agreement.

    This deliberately avoids reference answers. MATH-500 is handled by the
    separate pairwise symbolic-equivalence clustering path because unsafe
    string normalization would split mathematically equivalent answers.
    """

    name = dataset.lower()
    if name in {"arc_challenge", "mmlu_pro", "truthfulqa", "gpqa_diamond"}:
        return extract_choice(output_text)
    if name in {"gsm8k", "synthetic"}:
        value = extract_number(output_text)
        if value is None:
            return None
        if float(value).is_integer():
            return str(int(value))
        return format(float(value), ".12g")
    if name == "math500":
        raise ValueError(
            "MATH-500 requires pairwise symbolic clustering; use "
            "math500_equivalent() rather than string canonicalization."
        )
    raise ValueError(f"No canonical prediction parser for dataset: {dataset}")
