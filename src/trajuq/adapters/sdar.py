"""SDAR upstream integration boundary."""

UPSTREAM_REPOSITORY = "JetAstra/SDAR"
UPSTREAM_COMMIT = "4c2749ba103448f45520e8411533710a1e66574d"


def integration_status() -> dict[str, str]:
    return {
        "status": "not_yet_parity_validated",
        "repository": UPSTREAM_REPOSITORY,
        "commit": UPSTREAM_COMMIT,
        "hook": "observe logits before transfer and token state after each denoising step",
    }
