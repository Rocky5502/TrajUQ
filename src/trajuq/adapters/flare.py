"""FLARE / HybridDiffusion upstream integration boundary."""

UPSTREAM_REPOSITORY = "yuchen-zhu-zyc/HybridDiffusion"
UPSTREAM_COMMIT = "6ca547aebb72bfe897e80e0a683c776789e5f38c"


def integration_status() -> dict[str, str]:
    return {
        "status": "not_yet_parity_validated",
        "repository": UPSTREAM_REPOSITORY,
        "commit": UPSTREAM_COMMIT,
        "diffusion_hook": "emit pre-commit logits/state summaries per denoising step",
        "ar_diagnostic": "emit final AR answer/confidence separately; never treat it as ground truth",
    }
