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
        new_symbol,
        depth: int,
        base_type,
        context: Dict[str, str],
    ):
        new_string = ""
        for letter in [(["a", "t"], [0.09, 0.91])]:
            r.choice_weighted(letter[0], letter[1])
        rec(new_string)
        #rec(r.randint(self.min, self.max))

    def __repr__(self):
        return f"[{self.min}...{self.max}]"
