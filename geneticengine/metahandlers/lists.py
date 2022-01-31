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
        ctx: Dict[str, str],
    ):
        
        size = r.randint(0, depth - 1)
        fins = build_finalizers(lambda *x: rec(list(x)), size)
        ident = ctx["_"]
        for i, fin in enumerate(fins):
            nctx = ctx.copy()
            nctx["_"] = ident + "_" + str(i)
            new_symbol(base_type, fin, depth - 1, ident, nctx)

    def __repr__(self):
        return f"ListSizeBetween[{self.min}...{self.max}]"
