from dataclasses import asdict

from trajuq.schema import ExampleRecord, GenerationResult, StepSummary


def test_raw_record_has_no_outcome():
    generation = GenerationResult(
        text="FINAL: 1",
        token_ids=[],
        features={"entropy_final": 0.1},
        steps=[
            StepSummary(
                step=0,
                entropy_mean=0.1,
                entropy_p90=0.1,
                confidence_mean=0.9,
                margin_mean=0.8,
                mask_fraction=0.0,
                revision_fraction=0.0,
            )
        ],
        metadata={"model_revision": "x"},
    )
    record = ExampleRecord(
        example_id="x",
        dataset="synthetic",
        prompt_sha256="abc",
        model="toy",
        adapter="synthetic",
        seed=1,
        generation=generation,
    )
    payload = asdict(record)
    assert payload["correct"] is None
    assert payload["reference"] is None
