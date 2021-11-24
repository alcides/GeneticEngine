from random import Random
from typing import Protocol, TypeVar, ForwardRef, Tuple, get_args

from geneticengine.core.random.sources import RandomSource
from geneticengine.core.utils import build_finalizers
from geneticengine.metahandlers.base import MetaHandlerGenerator

min = TypeVar("min", covariant=True)
max = TypeVar("max", covariant=True)


class ListSizeBetween(MetaHandlerGenerator):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def generate(self, r: RandomSource, receiver, new_symbol, depth, base_type):
        size = r.randint(self.min, self.max)
        fins = build_finalizers(lambda *x: receiver(x), size)
        for i in range(size):
            new_symbol(base_type, fins[i], depth - 1)

    def __repr__(self):
        return f"ListSizeBetween[{self.min}...{self.max}]"
