import pytest

from trajuq.decision import Action, DecisionPolicy


def test_three_way_policy():
    p = DecisionPolicy(accept_max_error_prob=0.2, abstain_min_error_prob=0.7)
    assert p.decide(0.1) == Action.ACCEPT
    assert p.decide(0.4) == Action.MORE_COMPUTE
    assert p.decide(0.9) == Action.ABSTAIN


def test_invalid_thresholds():
    with pytest.raises(ValueError):
        DecisionPolicy(0.8, 0.7)
