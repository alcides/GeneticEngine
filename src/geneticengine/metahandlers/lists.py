from random import Random
from typing import Protocol, TypeVar, ForwardRef, Tuple, get_args

from geneticengine.core.random.sources import RandomSource

min = TypeVar("min", covariant=True)
max = TypeVar("max", covariant=True)


class ListSizeBetween(object):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def generate(self, r: RandomSource, recursive_generator):
        size = r.randint(self.min, self.max)
        return [recursive_generator() for _ in range(size)]

    def __repr__(self):
        return f"ListSizeBetween[{self.min}...{self.max}]"