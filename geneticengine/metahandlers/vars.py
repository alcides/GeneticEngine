from typing import (
    Any,
    Callable,
    Generic,
    Protocol,
    TypeVar,
    ForwardRef,
    Tuple,
    get_args,
    Dict,
    Type,
)

from geneticengine.core.random.sources import RandomSource
from geneticengine.metahandlers.base import MetaHandlerGenerator

from geneticengine.core.grammar import Grammar


class VarRange(MetaHandlerGenerator):
    def __init__(self, options):
        self.options = options

    def generate(
        self,
        r: RandomSource,
        g: Grammar,
        wrapper: Callable[[Any, str, int, Callable[[int], Any]], Any],
        rec: Any,
        depth: int,
        base_type,
        argname: str,
        context: Dict[str, Type],
    ):
        return r.choice(self.options)

    def __repr__(self):
        return str(self.options)
