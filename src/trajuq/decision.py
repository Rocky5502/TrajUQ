from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    ACCEPT = "accept"
    MORE_COMPUTE = "more_compute"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class DecisionPolicy:
    accept_max_error_prob: float = 0.20
    abstain_min_error_prob: float = 0.70

    def __post_init__(self) -> None:
        if not 0.0 <= self.accept_max_error_prob <= 1.0:
            raise ValueError("accept_max_error_prob must be in [0, 1]")
        if not 0.0 <= self.abstain_min_error_prob <= 1.0:
            raise ValueError("abstain_min_error_prob must be in [0, 1]")
        if self.accept_max_error_prob >= self.abstain_min_error_prob:
            raise ValueError("accept threshold must be below abstain threshold")

    def decide(self, error_probability: float) -> Action:
        p = float(error_probability)
        if not 0.0 <= p <= 1.0:
            raise ValueError("error_probability must be in [0, 1]")
        if p <= self.accept_max_error_prob:
            return Action.ACCEPT
        if p >= self.abstain_min_error_prob:
            return Action.ABSTAIN
        return Action.MORE_COMPUTE
