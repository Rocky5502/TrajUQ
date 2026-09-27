from scripts.preflight import MIN_TRANSFORMERS_VERSION, version_triplet


def test_version_triplet_parses_release_versions():
    assert version_triplet("4.52.4") == (4, 52, 4)
    assert version_triplet("4.57.1") == (4, 57, 1)


def test_version_triplet_accepts_suffix_after_patch():
    assert version_triplet("4.52.4.dev0") == (4, 52, 4)
    assert version_triplet("4.52.4rc1") == (4, 52, 4)


def test_transformers_floor_contract():
    assert version_triplet("4.52.3") < MIN_TRANSFORMERS_VERSION
    assert version_triplet("4.52.4") >= MIN_TRANSFORMERS_VERSION
    assert version_triplet("4.53.0") >= MIN_TRANSFORMERS_VERSION


def test_version_triplet_rejects_unparseable_value():
    assert version_triplet("unknown") is None
