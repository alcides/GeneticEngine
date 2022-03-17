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
    """
        ListSizeBetween(a,b) restricts lists to be of length between a and b.
        The list of options can be dynamically altered before the grammar extraction (Set.__annotations__["set"] = Annotated[List[Type], ListSizeBetween(c,d)].
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
        ctx: Dict[str, str],
    ):
        base_type = base_type.__args__[0]
        size = r.randint(self.min, self.max)
        fins = build_finalizers(lambda *x: rec(list(x)), size)
        ident = ctx["_"]
        for i, fin in enumerate(fins):
            nctx = ctx.copy()
            nident = ident + "_" + str(i)
            nctx["_"] = nident
            new_symbol(base_type, fin, depth - 1, nident, nctx)

    def __repr__(self):
        return f"ListSizeBetween[{self.min}...{self.max}]"
