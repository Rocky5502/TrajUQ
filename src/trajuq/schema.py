from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = "trajuq-record-v3"


@dataclass
class StepSummary:
    step: int
    entropy_mean: float
    entropy_p90: float
    confidence_mean: float
    margin_mean: float
    mask_fraction: float
    revision_fraction: float
    js_mean: float = 0.0
    active_fraction: float = 1.0
    active_count: int = 0


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]
    features: dict[str, float]
    steps: list[StepSummary]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExampleRecord:
    example_id: str
    dataset: str
    prompt_sha256: str
    model: str
    adapter: str
    seed: int
    generation: GenerationResult
    correct: bool | None = None
    reference: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
