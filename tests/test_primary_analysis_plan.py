from scripts.plan_primary_analysis import main as analysis_main


def test_primary_analysis_planner_imports():
    assert callable(analysis_main)
