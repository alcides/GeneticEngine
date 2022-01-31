from typing import (
    Any,
    Callable,
    TypeVar,
    Dict,
    Type,
)

from geneticengine.core.random.sources import Source
from geneticengine.core.utils import build_finalizers
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
        rec,
        new_symbol,
        depth: int,
        base_type,
        context: Dict[str, str],
    ):
        size = r.randint(self.min, self.max)
        fins = build_finalizers(lambda *x: rec(list(x)), size)
        for i in range(size):
            newsymbol(base_type, fins[i], depth - 1, context)

    def __repr__(self):
        return f"ListSizeBetween[{self.min}...{self.max}]"
