from random import Random
from typing import Protocol, TypeVar, ForwardRef, Tuple, get_args

from geneticengine.core.random.sources import RandomSource

min = TypeVar("min", covariant=True)
max = TypeVar("max", covariant=True)


class IntRange(object):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def generate(self, r: RandomSource):
        return r.randint(self.min, self.max)

    def __repr__(self):
        return f"[{self.min}...{self.max}]"