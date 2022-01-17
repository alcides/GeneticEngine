from random import Random
from typing import (
    Any,
    Callable,
    Protocol,
    TypeVar,
    ForwardRef,
    Tuple,
    get_args,
    Dict,
    Type,
)

from geneticengine.core.random.sources import RandomSource
from geneticengine.metahandlers.base import MetaHandlerGenerator

from geneticengine.core.grammar import Grammar

min = TypeVar("min", covariant=True)
max = TypeVar("max", covariant=True)


class ListSizeBetween(MetaHandlerGenerator):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def generate(
        self,
        r: RandomSource,
        g: Grammar,
        wrapper: Callable[[Any, str, int, Callable[[int], Any]], Any],
        rec: Any,
        depth: int,
        base_type,
        argname: str,
        context: Dict[str, Type],
    ):
        size = r.randint(self.min, self.max)
        return [rec(r, g, wrapper, depth - 1, base_type) for _ in range(size)]

    def __repr__(self):
        return f"ListSizeBetween[{self.min}...{self.max}]"
