from typing import (
    Any,
    Callable,
    Dict,
    List,
    Type,
    TypeVar,
)

from geneticengine.core.random.sources import Source
from geneticengine.metahandlers.base import MetaHandlerGenerator

from geneticengine.core.grammar import Grammar

T = TypeVar("T")

class VarRange(MetaHandlerGenerator):
    """
        VarRange([a, b, c]) represents the alternative between a, b, and c.
        The list of options can be dynamically altered before the grammar extraction
        with something like Var.__init__.__annotations__["name"] = Annotated[str, VarRange([d, e, f])].
        The option list must not be empty.
    """
    def __init__(self, options:List[T]):
        if not options:
            raise Exception(f"The VarRange metahandler requires a non-empty set of options. Options found: {options}")
        self.options = options

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
        rec(r.choice(self.options))

    def __repr__(self):
        return str(self.options)
