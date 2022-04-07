from __future__ import annotations

import random
from abc import ABC
from itertools import accumulate
from typing import Any
from typing import List
from typing import TypeVar

T = TypeVar("T")


class Source(ABC):
    def __init__(self, seed: int = 0):
        ...

    def randint(self, min: int, max: int) -> int:
        ...

    def random_float(self, min: float, max: float) -> float:
        ...

    def choice(self, choices: List[T]) -> T:
        assert choices
        i = self.randint(0, len(choices) - 1)
        return choices[i]

    def choice_weighted(self, choices: list[T], weights: list[float]) -> T:
        acc_weights = list(accumulate(weights))
        total = acc_weights[-1] + 0.0
        rand_value: float = self.random_float(0, total)

        for (choice, acc) in zip(choices, acc_weights):
            if rand_value < acc:
                return choice
        return choices[0]

    def shuffle(self, l: list[T]):
        for i in reversed(range(1, len(l))):
            j = self.randint(0, i)
            l[i], l[j] = l[j], l[i]
        return l

    def pop_random(self, l: list[T]) -> T:
        item = l.pop()
        total_len = len(l)

        i = self.randint(0, total_len)
        if i == total_len:
            return item

        l[i], item = item, l[i]

        return item

    def random_bool(self) -> bool:
        return self.choice([True, False])


class RandomSource(Source):
    def __init__(self, seed: int = 0):
        self.random = random.Random(seed)

    def randint(self, min, max) -> int:
        return self.random.randint(min, max)

    def random_float(self, min, max) -> float:
        return self.random.random() * (max - min) + min
