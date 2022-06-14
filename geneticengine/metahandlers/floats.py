from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import Type
from typing import TypeVar

from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import Source
from geneticengine.metahandlers.base import MetaHandlerGenerator

min = TypeVar("min", covariant=True)
max = TypeVar("max", covariant=True)


class FloatRange(MetaHandlerGenerator):
    """
    FloatRange(a,b) restricts floats to be between a and b.
    The range can be dynamically altered before the grammar extraction (Float.__annotations__["value"] = Annotated[float, FloatRange(c,d)].
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def generate(
        self,
        r: Source,
        g: Grammar,
        rec,
        new_symbol,
        depth: int,
        base_type,
        context: dict[str, str],
    ):
        rec(r.random_float(self.min, self.max, str(base_type)))

    def __class_getitem__(self, args):
        return FloatRange(*args)

    def __repr__(self):
        return f"[{self.min},{self.max}]"


class FloatList(MetaHandlerGenerator):
    """
    FloatList([a_1, .., a_n]) restricts floats to be an element from the list [a_1, .., a_n].
    The range can be dynamically altered before the grammar extraction (Float.__init__.__annotations__["value"] = Annotated[float, FloatList[a_1, .., a_n]].
    """

    def __init__(self, elements):
        self.elements = elements

    def generate(
        self,
        r: Source,
        g: Grammar,
        rec,
        new_symbol,
        depth: int,
        base_type,
        context: dict[str, str],
    ):
        rec(r.choice(self.elements, str(base_type)))

    def __class_getitem__(self, args):
        return FloatList(*args)

    def __repr__(self):
        return f"[{self.elements}]"
