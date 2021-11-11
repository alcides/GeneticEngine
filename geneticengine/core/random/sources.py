from abc import ABC
import random

from typing import List, Any, Protocol


class Source(ABC):
    def __init__(self, seed: int = 0):
        ...

    def randint(self, min, max) -> int:
        ...

    def random_float(self, min, max) -> float:
        ...

    def choice(self, choices: List[Any]) -> Any:
        assert choices
        i = self.randint(0, len(choices) - 1)
        return choices[i]


class RandomSource(Source):
    def __init__(self, seed: int = 0):
        self.random = random.Random(seed)

    def randint(self, min, max) -> int:
        return self.random.randint(min, max)

    def random_float(self, min, max) -> float:
        return self.random.random() * (max - min) + min
