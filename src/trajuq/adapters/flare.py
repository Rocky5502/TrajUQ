"""HybridDiffusion/FLARE integration metadata.

The pinned HybridDiffusion SGLang runtime exposes a native JSONL trace hook.
TrajUQ consumes that trace externally rather than vendoring or modifying the
specialized server implementation.
"""

UPSTREAM_REPOSITORY = "yuchen-zhu-zyc/HybridDiffusion"
UPSTREAM_COMMIT = "6ca547aebb72bfe897e80e0a683c776789e5f38c"
TRACE_PATH_ENV = "SGLANG_HYBRID_DIFFUSION_SELF_SPEC_TRACE_PATH"
TRACE_TOPK_ENV = "SGLANG_HYBRID_DIFFUSION_SELF_SPEC_TRACE_TOPK"


def integration_status() -> dict[str, str]:
    return {
        "status": "native_trace_bridge_implemented_parity_pending",
        "repository": UPSTREAM_REPOSITORY,
        "commit": UPSTREAM_COMMIT,
        "trace_path_env": TRACE_PATH_ENV,
        "trace_topk_env": TRACE_TOPK_ENV,
        "bridge": "scripts/convert_hybrid_trace.py",
        "entropy_note": "native trace supports a top-k entropy proxy, not full-vocabulary entropy",
        "ar_diagnostic": "record AR output separately; never treat it as correctness ground truth",
    }
