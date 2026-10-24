from pathlib import Path


def test_registry_is_freeze_bound_uuid_validated_and_artifact_driven():
    text = Path("scripts/build_rq5_validated_registry.py").read_text(encoding="utf-8")
    assert "uuid.uuid5" in text
    assert "score_record" in text
    assert "duplicate successful request mapping" in text
    assert "generated_tex_sha256" in text
    assert '"partial": len(validated) < expected_total' in text
    assert "RQ5 calibration and uncertainty ranking" in text
    assert "Provisional RQ5 operational accounting" in text
