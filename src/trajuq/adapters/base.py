from __future__ import annotations

from abc import ABC, abstractmethod

from trajuq.schema import GenerationResult


class BaseAdapter(ABC):
    name: str

    @abstractmethod
    def generate(self, prompt: str, seed: int = 2027, **kwargs) -> GenerationResult:
        raise NotImplementedError
