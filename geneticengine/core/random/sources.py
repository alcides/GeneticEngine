from __future__ import annotations

import random
from abc import ABC, abstractmethod
from itertools import accumulate
from typing import Any
from typing import List
from typing import TypeVar

from geneticengine.core.representations.tree.utils import GengyList
from geneticengine.core.utils import build_finalizers
from geneticengine.core.utils import get_generic_parameter

T = TypeVar("T")


class Source(ABC):
    @abstractmethod
    def __init__(self, seed: int = 0):
        ...

    @abstractmethod
    def randint(self, min: int, max: int, prod: str = "") -> int:
        ...

    @abstractmethod
    def random_float(self, min: float, max: float, prod: str = "") -> float:
        ...

    def choice(self, choices: list[T], prod: str = "") -> T:
        assert choices
        i = self.randint(0, len(choices) - 1, prod)
        return choices[i]

    def choice_weighted(
        self,
        choices: list[T],
        weights: list[float],
        prod: str = "",
    ) -> T:
        acc_weights = list(accumulate(weights))
        total = acc_weights[-1] + 0.0
        rand_value: float = self.random_float(0, total, prod)

        for (choice, acc) in zip(choices, acc_weights):
            if rand_value < acc:
                return choice
        return choices[0]

    def shuffle(self, l: list[T], prod: str = ""):
        for i in reversed(range(1, len(l))):
            j = self.randint(0, i, prod)
            l[i], l[j] = l[j], l[i]
        return l

    def pop_random(self, l: list[T], prod: str = "") -> T:
        item = l.pop()
        total_len = len(l)

        i = self.randint(0, total_len, prod)
        if i == total_len:
            return item

        l[i], item = item, l[i]

        return item

    def random_bool(self, prod: str = "") -> bool:
        return self.choice([True, False], prod)

    def random_list(
        self,
        receiver,
        new_symbol,
        depth: int,
        ty: type[list[T]],
        ctx: dict[str, str],
        prod: str = "",
    ):
        inner_type = get_generic_parameter(ty)
        size = 1
        if depth > 0:
            size = self.randint(1, depth, prod)
        fins = build_finalizers(lambda *x: receiver(GengyList(inner_type, x)), size)
        ident = ctx["_"]
        for i, fin in enumerate(fins):
            nctx = ctx.copy()
            nident = ident + "_" + str(i)
            nctx["_"] = nident
            new_symbol(inner_type, fin, depth, nident, nctx)


class RandomSource(Source):
    def __init__(self, seed: int = 0):
        self.random = random.Random(seed)

    def randint(self, min, max, prod: str = "") -> int:
        return self.random.randint(min, max)

    def random_float(self, min, max, prod: str = "") -> float:
        return self.random.random() * (max - min) + min
