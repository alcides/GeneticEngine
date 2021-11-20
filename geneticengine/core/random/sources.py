from abc import ABC
import random
from itertools import accumulate

from typing import List, Any, Protocol


class Source(ABC):
    def __init__(self, seed: int = 0):
        ...

    def randint(self, min: int, max: int) -> int:
        ...

    def random_float(self, min: float, max: float) -> float:
        ...

    def choice(self, choices: List[Any]) -> Any:
        assert choices
        i = self.randint(0, len(choices) - 1)
        return choices[i]

    def choice_weighted(self, choices: List[Any], weights: List[float]) -> Any:
        acc_weights = list(accumulate(weights))
        total = acc_weights[-1] + 0.0
        rand_value: float = self.random_float(0, total)

        for (choice, acc) in zip(choices, acc_weights):
            if rand_value < acc:
                return choice


class RandomSource(Source):
    def __init__(self, seed: int = 0):
        self.random = random.Random(seed)

    def randint(self, min, max) -> int:
        return self.random.randint(min, max)

    def random_float(self, min, max) -> float:
        return self.random.random() * (max - min) + min
