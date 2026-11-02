from pathlib import Path


def test_rq5_plan_is_chronology_honest_and_network_free():
    text = Path("scripts/freeze_rq5_downstream_plan.py").read_text(encoding="utf-8")
    assert "POST_DATA_PRE_OUTCOME_ANALYSIS" in text
    assert '"network_calls_allowed": False' in text
    assert "NOT_IDENTIFIABLE if model identity is unmatched" in text
    assert "Holm-Bonferroni over estimable confirmatory p-values" in text


def test_rq5_pipeline_is_fail_closed_and_partitioned():
    text = Path("scripts/rq5_downstream_stage.py").read_text(encoding="utf-8")
    assert 'with path.open("xb")' in text
    assert 'row["split"] == "calibration"' in text
    assert 'row["split"] != "test"' in text
    assert "paired_class_stratified_within_cell_equal_cell_macro" in text
    assert "NOT_IDENTIFIABLE_MODELS_UNMATCHED_ACROSS_OBSERVABILITY_REGIMES" in text
    assert "urlopen" not in text
    assert "import requests" not in text
    assert "from requests" not in text
