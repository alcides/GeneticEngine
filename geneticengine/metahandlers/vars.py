from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Type
from typing import TypeVar

from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import Source
from geneticengine.metahandlers.base import MetaHandlerGenerator

T = TypeVar("T")


class VarRange(MetaHandlerGenerator):
    """
    VarRange([a, b, c]) represents the alternative between a, b, and c.
    The list of options can be dynamically altered before the grammar extraction
    with something like Var.__init__.__annotations__["name"] = Annotated[str, VarRange([d, e, f])].
    The option list must not be empty.
    """

    def __init__(self, options: list[T]):
        if not options:
            raise Exception(
                f"The VarRange metahandler requires a non-empty set of options. Options found: {options}",
            )
        self.options = options

    def generate(
        self,
        r: Source,
        g: Grammar,
        rec,
        new_symbol,
        depth: int,
        base_type,
        context: dict[str, str],
    ):
        rec(r.choice(self.options))

    def __repr__(self):
        return str(self.options)

    def __class_getitem__(self, args):
        return VarRange(*args)
