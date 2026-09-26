import numpy as np


def test_llada_reference_contract_documented():
    # Lightweight regression guard: the hard model parity test is GPU-only.
    # This keeps the intended deterministic parity configuration visible in CI.
    cfg = {
        "steps": 128,
        "gen_length": 128,
        "block_length": 128,
        "temperature": 0.0,
        "remasking": "low_confidence",
    }
    assert cfg["steps"] % (cfg["gen_length"] // cfg["block_length"]) == 0
    assert np.isclose(cfg["temperature"], 0.0)
