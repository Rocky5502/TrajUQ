from scripts.plan_adaptive_runs import replace_arg


def test_replace_arg_updates_existing_value():
    argv = ["python", "x.py", "--steps", "128", "--output", "old"]
    out = replace_arg(argv, "--steps", "32")
    assert out == ["python", "x.py", "--steps", "32", "--output", "old"]
    assert argv[3] == "128"


def test_replace_arg_appends_missing_flag():
    assert replace_arg(["python", "x.py"], "--steps", "32")[-2:] == ["--steps", "32"]
