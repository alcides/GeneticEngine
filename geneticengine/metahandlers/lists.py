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
        depth,
        base_type,
        argname: str,
        context: Dict[str, Type],
    ):
        size = r.randint(self.min, self.max)
        fins = build_finalizers(lambda *x: rec(list(x)), size)
        for i in range(size):
            new_symbol(base_type, fins[i], depth - 1)

    def __repr__(self):
        return f"ListSizeBetween[{self.min}...{self.max}]"
