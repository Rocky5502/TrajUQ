import pytest

from trajuq.scoring import (
    extract_choice,
    extract_number,
    math500_equivalent,
    score_record,
)


def test_extractors():
    assert extract_choice("Reasoning... FINAL: C") == "C"
    assert extract_number("Reasoning... FINAL: 1,234.5") == 1234.5


def test_gsm8k_scoring():
    ok, name = score_record("gsm8k", "work\nFINAL: 18", "some work\n#### 18")
    assert ok
    assert name == "gsm8k_numeric_exact"


def test_mc_scoring():
    assert score_record("mmlu_pro", "FINAL: B", "B")[0]
    assert not score_record("mmlu_pro", "FINAL: C", "B")[0]


def test_math500_uses_symbolic_scorer_when_dependency_is_installed():
    pytest.importorskip("math_verify")
    ok, name = score_record("math500", r"FINAL: \frac{1}{2}", r"\frac{1}{2}")
    assert ok
    assert name == "math_verify_0.9.0_precision6"



def test_math500_symbolic_equivalence_for_repeated_sampling():
    pytest.importorskip("math_verify")
    assert math500_equivalent(r"FINAL: \frac{2}{4}", r"FINAL: 0.5")
    assert not math500_equivalent(r"FINAL: \frac{1}{2}", r"FINAL: \frac{3}{4}")



def test_math500_reference_is_boxed_independently_of_prediction_path():
    pytest.importorskip("math_verify")
    # The prediction includes surrounding text and an explicit FINAL field while
    # the benchmark reference is a bare expression. This guards the scorer's
    # asymmetric extraction contract.
    ok, _ = score_record(
        "math500",
        r"Reasoning that should not be graded. FINAL: \frac{2}{4}",
        r"\frac{1}{2}",
    )
    assert ok
