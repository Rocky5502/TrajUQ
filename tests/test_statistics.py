import numpy as np

from trajuq.statistics import paired_bootstrap_difference, paired_bootstrap_improvement


def test_paired_bootstrap_prefers_better_ranking():
    y = np.array([0, 0, 0, 1, 1, 1])
    good = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    bad = np.array([0.6, 0.2, 0.7, 0.1, 0.8, 0.4])
    result = paired_bootstrap_difference(y, good, bad, "auroc", n_boot=100, seed=1)
    assert result.point > 0
    assert result.valid_boot == 100



def test_aurc_improvement_direction_is_positive_for_better_ranking():
    y = np.array([0, 0, 0, 1, 1, 1])
    good = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    bad = np.array([0.6, 0.2, 0.7, 0.1, 0.8, 0.4])
    result = paired_bootstrap_improvement(
        y,
        good,
        bad,
        "aurc",
        n_boot=100,
        seed=1,
    )
    assert result.point > 0
