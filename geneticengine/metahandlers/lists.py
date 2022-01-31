from typing import Any, Callable, TypeVar, Dict, Type, List

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
        newsymbol,
        depth: int,
        base_type,
        context: Dict[str, str],
    ):
        base_type = base_type.__args__[0]
        size = r.randint(self.min, self.max)
        fins = build_finalizers(lambda *x: rec(list(x)), size)
        ident = context["_"]
        for i in range(size):
            nctx = context.copy()
            nctx["_"] = ident + "_" + str(i)
            newsymbol(base_type, fins[i], depth - 1, ident, context)

    def __repr__(self):
        return f"ListSizeBetween[{self.min}...{self.max}]"
