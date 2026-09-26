from scripts.export_math500_scorer_review import choose_diverse


def test_choose_diverse_round_robins_subject_level_groups():
    rows = []
    for index, (subject, level) in enumerate(
        [
            ("algebra", "1"),
            ("algebra", "1"),
            ("geometry", "2"),
            ("geometry", "2"),
            ("number_theory", "3"),
        ]
    ):
        rows.append(
            {
                "id": str(index),
                "source": {
                    "metadata": {"subject": subject, "level": level}
                },
            }
        )

    selected = choose_diverse(rows, count=3, seed=2027)
    groups = {
        (
            item["source"]["metadata"]["subject"],
            item["source"]["metadata"]["level"],
        )
        for item in selected
    }
    assert len(selected) == 3
    assert len(groups) == 3
