from scripts.prepare_eval_pair import apply_frozen_caps


def _rows(prefix: str, count: int) -> list[dict]:
    return [{"id": f"{prefix}:{i}"} for i in range(count)]


def test_apply_frozen_caps_preserves_uncapped_counts_and_caps_once():
    cal = _rows("cal", 8)
    test = _rows("test", 11)

    capped_cal, capped_test, counts = apply_frozen_caps(
        cal,
        test,
        dataset="gsm8k",
        seed="2027",
        calibration_max=5,
        test_max=7,
    )

    assert counts == {
        "source_total": 19,
        "calibration_before_cap": 8,
        "test_before_cap": 11,
    }
    assert len(capped_cal) == 5
    assert len(capped_test) == 7


def test_apply_frozen_caps_keeps_uncapped_partition_full():
    cal = _rows("cal", 3)
    test = _rows("test", 4)

    capped_cal, capped_test, counts = apply_frozen_caps(
        cal,
        test,
        dataset="truthfulqa",
        seed="2027",
        calibration_max=None,
        test_max=None,
    )

    assert capped_cal == cal
    assert capped_test == test
    assert counts["source_total"] == 7
    assert counts["calibration_before_cap"] == 3
    assert counts["test_before_cap"] == 4


def test_apply_frozen_caps_is_deterministic_under_input_reordering():
    cal = _rows("cal", 20)
    test = _rows("test", 20)

    a_cal, a_test, _ = apply_frozen_caps(
        cal,
        test,
        dataset="mmlu_pro",
        seed="2027",
        calibration_max=6,
        test_max=6,
    )
    b_cal, b_test, _ = apply_frozen_caps(
        list(reversed(cal)),
        list(reversed(test)),
        dataset="mmlu_pro",
        seed="2027",
        calibration_max=6,
        test_max=6,
    )

    assert [row["id"] for row in a_cal] == [row["id"] for row in b_cal]
    assert [row["id"] for row in a_test] == [row["id"] for row in b_test]
