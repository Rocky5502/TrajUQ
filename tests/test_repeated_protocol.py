import pytest

from trajuq.repeated_sampling import vote_uncertainty


def test_five_seed_protocol_has_expected_size():
    seeds = [2027, 2028, 2029, 2030, 2031]
    assert len(seeds) == 5
    assert len(seeds) == len(set(seeds))


def test_vote_uncertainty_requires_multiple_samples():
    with pytest.raises(ValueError):
        vote_uncertainty(["A"])
