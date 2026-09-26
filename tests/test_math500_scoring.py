import pytest

pytest.importorskip("math_verify")

from trajuq.scoring import score_math500


@pytest.mark.parametrize(
    ("prediction", "gold"),
    [
        (r"FINAL: \frac{1}{2}", r"\frac{1}{2}"),
        (r"FINAL: 0.5", r"\frac{1}{2}"),
        (r"FINAL: x^2 + 2*x + 1", r"(x+1)^2"),
        (r"The computation gives $3/4$. FINAL: \boxed{\frac{3}{4}}", r"\frac{3}{4}"),
        (r"FINAL: 2", r"\sqrt{4}"),
    ],
)
def test_math500_symbolic_equivalence(prediction, gold):
    assert score_math500(prediction, gold)


def test_math500_incorrect_answer_rejected():
    assert not score_math500(r"FINAL: \frac{2}{3}", r"\frac{1}{2}")
