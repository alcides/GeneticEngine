from __future__ import annotations

import abc
import math
import random
from itertools import accumulate
from typing import TypeVar

from geneticengine.core.representations.tree.utils import GengyList
from geneticengine.core.utils import build_finalizers
from geneticengine.core.utils import get_generic_parameter

T = TypeVar("T")


class Source(abc.ABC):
    @abc.abstractmethod
    def randint(self, min: int, max: int, prod: str = "") -> int:
        ...

    @abc.abstractmethod
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
        acc_weights: list[int] = [int(x * 100000) for x in accumulate(weights)]
        total = acc_weights[-1]
        rand_value: float = self.randint(0, total, prod)

        for (choice, acc) in zip(choices, acc_weights):
            if rand_value < acc:
                return choice
        return choices[0]

    def shuffle(self, lst: list[T]):
        for i in reversed(range(1, len(lst))):
            j = self.randint(0, i)
            lst[i], lst[j] = lst[j], lst[i]
        return lst

    def pop_random(self, lst: list[T]) -> T:
        item = lst.pop()
        total_len = len(lst)

        i = self.randint(0, total_len)
        if i == total_len:
            return item

        lst[i], item = item, lst[i]

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

    def normalvariate(
        self,
        mean: float,
        sigma: float,
        prod: str,
    ) -> float:
        # Box-Muller transform https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
        # I also found this approach https://rh8liuqy.github.io/Box_Muller_Algorithm.html using numpy library instead of math library
        u1 = self.random_float(0.0, 1.0, prod)
        u2 = self.random_float(0.0, 1.0, prod)
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)
        return z0 * sigma + mean


class RandomSource(Source):
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.random = random.Random(seed)

    def normalvariate(self, mean, sigma, prod: str = "") -> float:
        return self.random.normalvariate(mean, sigma)

    def randint(self, min, max, prod: str = "") -> int:
        return self.random.randint(min, max)

    def random_float(self, min, max, prod: str = "") -> float:
        return self.random.random() * (max - min) + min
