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

min = TypeVar("min", covariant=True)
max = TypeVar("max", covariant=True)


class IntRange(MetaHandlerGenerator):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def generate(
        self,
        r: Source,
        g: Grammar,
        rec,
        newsymbol,
        depth: int,
        base_type,
        context: Dict[str, str],
    ):
        rec(r.randint(self.min, self.max))

    def __repr__(self):
        return f"[{self.min}...{self.max}]"
