from scripts.render_authorized_rq23_tex import validate_cap, validate_rq2


def test_rq2_authorization_requires_complete_registered_subset():
    value = {
        "protocol": "trajuq_vs_repeated_macro_v2",
        "comparison": "single_trajectory_trajuq_vs_five_sample_baseline",
        "cell_set_complete": True,
        "expected_cells": 12,
        "observed_cells": 12,
        "macro": {name: {} for name in ("auroc", "auprc", "aurc", "ece15", "nll")},
    }
    validate_rq2(value)


def test_adaptive_authorization_distinguishes_primary_and_sensitivity():
    value = {
        "protocol": "trajuq_adaptive_macro_v1",
        "execution_semantics": "selective_second_pass_rerun",
        "fraction_key": "cap_050",
        "max_refine_fraction": 0.5,
        "is_primary_fraction": True,
        "cell_set_complete": True,
        "expected_cells": 18,
        "observed_cells": 18,
    }
    validate_cap(value, "cap_050", 0.5, True)
