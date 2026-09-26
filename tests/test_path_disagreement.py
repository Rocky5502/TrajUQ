import math

from trajuq.path_disagreement import compare_paths


def test_mc_answer_disagreement():
    result = compare_paths(
        "x",
        "mmlu_pro",
        "Reasoning... FINAL: B",
        "Reasoning... FINAL: C",
        [1, 2, 3],
        [1, 2, 4],
    )
    assert result.features["exact_text_disagreement"] == 1.0
    assert result.features["parsed_answer_available"] == 1.0
    assert result.features["parsed_answer_disagreement"] == 1.0
    assert math.isclose(result.features["normalized_token_edit_distance"], 1 / 3)
    assert math.isclose(result.features["token_common_prefix_fraction"], 2 / 3)


def test_gsm8k_equivalent_numeric_surface_answers_agree():
    result = compare_paths(
        "x",
        "gsm8k",
        "FINAL: 18",
        "FINAL: 18.0",
        None,
        None,
    )
    assert result.features["parsed_answer_available"] == 1.0
    assert result.features["parsed_answer_disagreement"] == 0.0
    assert result.features["token_metrics_available"] == 0.0
