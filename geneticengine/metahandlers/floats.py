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

from geneticengine.core.random.sources import Source
from geneticengine.metahandlers.base import MetaHandlerGenerator

from geneticengine.core.grammar import Grammar

min = TypeVar("min", covariant=True)
max = TypeVar("max", covariant=True)


class FloatRange(MetaHandlerGenerator):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def generate(
        self,
        r: Source,
        g: Grammar,
        wrapper: Callable[[Any, str, int, Callable[[int], Any]], Any],
        rec: Any,
        depth: int,
        base_type,
        argname: str,
        context: Dict[str, Type],
    ):
        return r.random_float(self.min, self.max)

    def __repr__(self):
        return f"[{self.min}...{self.max}]"
