from scripts.plan_repeated_sampling import arg_value, repeated_generation_argv


def llada_cell():
    return {
        "adapter": "llada",
        "model_id": "m",
        "model_revision": "sha",
        "calibration": {
            "input": "cal.jsonl",
            "generate_argv": [
                "python",
                "scripts/run_generation.py",
                "--steps",
                "128",
                "--max-new-tokens",
                "128",
                "--block-length",
                "128",
                "--llada-cfg-scale",
                "0.0",
                "--llada-remasking",
                "low_confidence",
            ],
        },
    }


def test_arg_value_reads_frozen_flag():
    assert arg_value(["x", "--steps", "128"], "--steps") == "128"


def test_repeated_plan_uses_all_frozen_seeds_and_nonzero_temperature():
    cmd = repeated_generation_argv(
        llada_cell(),
        "calibration",
        "out",
        [2027, 2028, 2029, 2030, 2031],
        0.2,
        1.0,
    )
    assert cmd[cmd.index("--temperature") + 1] == "0.2"
    seed_idx = cmd.index("--seeds")
    assert cmd[seed_idx + 1 : seed_idx + 6] == [
        "2027",
        "2028",
        "2029",
        "2030",
        "2031",
    ]
    assert cmd[cmd.index("--steps") + 1] == "128"
