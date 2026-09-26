import math

from trajuq.hybrid_trace import summarize_hybrid_request, topk_entropy_proxy


def test_topk_entropy_proxy_bounds():
    concentrated = topk_entropy_proxy([10.0, 0.0, -1.0])
    flat = topk_entropy_proxy([1.0, 1.0, 1.0])
    assert concentrated is not None
    assert flat is not None
    assert 0.0 <= concentrated < flat <= 1.0
    assert math.isclose(flat, 1.0, abs_tol=1e-12)


def test_hybrid_summary_tracks_acceptance_correction_and_late_signal():
    rows = [
        {
            "req_pool_idx": 7,
            "forward_id": 1,
            "batch_idx": 0,
            "output_pos": 0,
            "source": "accepted_spec",
            "accepted": True,
            "argmax_margin": 4.0,
            "top_logits": [8.0, 4.0, 1.0],
        },
        {
            "req_pool_idx": 7,
            "forward_id": 1,
            "batch_idx": 0,
            "output_pos": 1,
            "source": "clean_sample",
            "accepted": True,
            "argmax_margin": 3.0,
            "top_logits": [7.0, 4.0, 1.0],
        },
        {
            "req_pool_idx": 7,
            "forward_id": 2,
            "batch_idx": 0,
            "output_pos": 0,
            "source": "accepted_spec",
            "accepted": True,
            "argmax_margin": 1.0,
            "top_logits": [3.0, 2.0, 1.0],
        },
        {
            "req_pool_idx": 7,
            "forward_id": 2,
            "batch_idx": 0,
            "output_pos": 1,
            "source": "correction",
            "accepted": False,
            "argmax_margin": 0.2,
            "top_logits": [1.2, 1.0, 0.8],
        },
    ]
    summary = summarize_hybrid_request("7", rows, late_fraction=0.5)
    f = summary.features
    assert f["trace_forwards"] == 2.0
    assert math.isclose(f["accepted_spec_fraction"], 2 / 3)
    assert f["reject_forward_fraction"] == 0.5
    assert f["late_correction_fraction"] == 0.5
    assert f["late_argmax_margin_mean"] < f["argmax_margin_mean"]
    assert f["late_topk_entropy_proxy_mean"] > 0.0
    assert summary.metadata["entropy_semantics"].startswith("topk_entropy_proxy")
