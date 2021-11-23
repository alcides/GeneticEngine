from random import Random
from typing import Protocol, TypeVar, ForwardRef, Tuple, get_args

from geneticengine.core.random.sources import RandomSource
from geneticengine.metahandlers.base import MetaHandlerGenerator

min = TypeVar("min", covariant=True)
max = TypeVar("max", covariant=True)


class FloatRange(MetaHandlerGenerator):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def generate(self, r: RandomSource, v):
        return r.random_float(self.min, self.max)

    def __repr__(self):
        return f"[{self.min}...{self.max}]"