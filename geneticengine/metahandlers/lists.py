from typing import (
    Any,
    Callable,
    TypeVar,
    Dict,
    Type,
)

from geneticengine.core.random.sources import Source
from geneticengine.metahandlers.base import MetaHandlerGenerator

from geneticengine.core.grammar import Grammar


class ListSizeBetween(MetaHandlerGenerator):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def generate(
        self,
        r: Source,
        g: Grammar,
        rec: Callable[[int, Type], Any],
        depth: int,
        base_type,
        argname: str,
        context: Dict[str, Type],
    ):
        size = r.randint(self.min, self.max)
        return [rec(depth - 1, base_type) for _ in range(size)]

    def __repr__(self):
        return f"ListSizeBetween[{self.min}...{self.max}]"
