from scripts.verify_dataset_contracts import verify_one


def fake_loader_factory(row):
    def load_dataset(repo_id, *args, **kwargs):
        assert repo_id
        assert kwargs["revision"] == "resolved-sha"
        assert kwargs["streaming"] is True
        assert kwargs["split"]
        return iter([row])

    return load_dataset


def specs_and_rows():
    return {
        "gsm8k": (
            {
                "hf_id": "openai/gsm8k",
                "config": "main",
                "strategy": "official",
                "calibration_split": "train",
            },
            {"question": "2+2?", "answer": "work #### 4"},
        ),
        "math500": (
            {
                "hf_id": "HuggingFaceH4/MATH-500",
                "config": None,
                "strategy": "internal_holdout",
                "source_split": "test",
            },
            {
                "problem": "Compute one half.",
                "answer": r"\frac{1}{2}",
                "subject": "Algebra",
                "level": 1,
            },
        ),
        "arc_challenge": (
            {
                "hf_id": "allenai/ai2_arc",
                "config": "ARC-Challenge",
                "strategy": "official",
                "calibration_split": "validation",
            },
            {
                "id": "arc-x",
                "question": "Pick two.",
                "choices": {
                    "text": ["one", "two", "three", "four"],
                    "label": ["A", "B", "C", "D"],
                },
                "answerKey": "B",
            },
        ),
        "mmlu_pro": (
            {
                "hf_id": "TIGER-Lab/MMLU-Pro",
                "config": None,
                "strategy": "official",
                "calibration_split": "validation",
            },
            {
                "question_id": 7,
                "question": "Pick B.",
                "options": ["one", "two", "three", "four"],
                "answer_index": 1,
                "category": "test",
            },
        ),
        "truthfulqa": (
            {
                "hf_id": "truthfulqa/truthful_qa",
                "config": "multiple_choice",
                "strategy": "internal_holdout",
                "source_split": "validation",
            },
            {
                "question": "Pick the true answer.",
                "mc1_targets": {
                    "choices": ["true", "false"],
                    "labels": [1, 0],
                },
            },
        ),
        "gpqa_diamond": (
            {
                "hf_id": "Idavidrein/gpqa",
                "config": "gpqa_diamond",
                "strategy": "internal_holdout",
                "source_split": "train",
            },
            {
                "Question": "Private synthetic question",
                "Correct Answer": "correct",
                "Incorrect Answer 1": "wrong1",
                "Incorrect Answer 2": "wrong2",
                "Incorrect Answer 3": "wrong3",
                "Subdomain": "synthetic",
            },
        ),
    }


def test_all_primary_normalizer_scorer_contracts_smoke():
    for name, (spec, row) in specs_and_rows().items():
        result = verify_one(
            name=name,
            spec=spec,
            revision="resolved-sha",
            load_dataset=fake_loader_factory(row),
        )
        assert result["status"] == "PASS", (name, result)
        assert result["scorer_smoke_pass"] is True
        assert result["source_row_content_stored"] is False
        assert result["resolved_revision"] == "resolved-sha"


def test_dataset_contract_artifact_exposes_schema_not_row_content():
    name = "gpqa_diamond"
    spec, row = specs_and_rows()[name]
    result = verify_one(
        name=name,
        spec=spec,
        revision="resolved-sha",
        load_dataset=fake_loader_factory(row),
    )

    rendered = str(result)
    assert "Private synthetic question" not in rendered
    assert "correct" not in rendered
    assert "wrong1" not in rendered
    assert set(result["upstream_fields"]) == set(row)
