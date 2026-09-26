from trajuq.repeated_sampling import vote_uncertainty


def test_vote_uncertainty_zero_when_all_samples_agree():
    result = vote_uncertainty(["A", "A", "A", "A", "A"])
    assert result["sample_disagreement"] == 0.0
    assert result["sample_vote_entropy"] == 0.0
    assert result["sample_unique_fraction"] == 0.2


def test_vote_uncertainty_increases_with_disagreement():
    agree = vote_uncertainty(["A", "A", "A", "A"])
    split = vote_uncertainty(["A", "A", "B", "B"])
    all_different = vote_uncertainty(["A", "B", "C", "D"])
    assert agree["sample_disagreement"] < split["sample_disagreement"]
    assert split["sample_disagreement"] < all_different["sample_disagreement"]
    assert agree["sample_vote_entropy"] < split["sample_vote_entropy"]
    assert split["sample_vote_entropy"] <= all_different["sample_vote_entropy"]
