from scripts.plan_adaptive_runs import fraction_key, replace_arg


def test_replace_arg_updates_existing_value():
    argv = ["python", "x.py", "--steps", "128", "--output", "old"]
    out = replace_arg(argv, "--steps", "32")
    assert out == ["python", "x.py", "--steps", "32", "--output", "old"]
    assert argv[3] == "128"


def test_replace_arg_appends_missing_flag():
    assert replace_arg(["python", "x.py"], "--steps", "32")[-2:] == ["--steps", "32"]



def test_fraction_key_is_stable():
    assert fraction_key(0.25) == "cap_025"
    assert fraction_key(0.50) == "cap_050"
    assert fraction_key(0.75) == "cap_075"
