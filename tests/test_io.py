from trajuq.io import sha256_text


def test_sha256_is_stable():
    assert sha256_text("TrajUQ") == sha256_text("TrajUQ")
    assert sha256_text("TrajUQ") != sha256_text("trajUQ")
