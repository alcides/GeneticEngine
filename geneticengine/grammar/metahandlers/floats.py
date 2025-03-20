from __future__ import annotations

from typing import Any, Callable, Generator, TypeVar

from geneticengine.grammar.grammar import Grammar
from geneticengine.random.sources import RandomSource
from geneticengine.grammar.metahandlers.base import MetaHandlerGenerator

min = TypeVar("min", covariant=True)
max = TypeVar("max", covariant=True)


T = TypeVar("T")


class FloatRange(MetaHandlerGenerator):
    """FloatRange(a,b) restricts floats to be between a and b.

    The range can be dynamically altered before the grammar extraction:
        Float.__annotations__["value"] = Annotated[float, FloatRange(c,d)].
    """

    min: float
    max: float

    def __init__(self, min: float, max: float):
        self.min = min
        self.max = max

    def generate(
        self,
        random: RandomSource,
        grammar: Grammar,
        base_type: type,
        rec: Callable[[type[T]], T],
        dependent_values: dict[str, Any],
        parent_values: list[dict[str, Any]],
    ):
        return random.random_float(self.min, self.max)

    def validate(self, v) -> bool:
        return self.min <= v <= self.max

    def __class_getitem__(cls, args):
        return FloatRange(*args)

    def __repr__(self):
        return f"[{self.min},{self.max}]"

    def iterate(
        self,
        base_type: type,
        combine_lists: Callable[[list[type]], Generator[Any, Any, Any]],
        rec: Any,
        dependent_values: dict[str, Any],
    ):
        v = self.min
        while v <= self.max:
            yield v
            v += 0.0001
