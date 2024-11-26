from __future__ import annotations

from typing import Any, Callable, Generator, TypeVar

from geneticengine.grammar.grammar import Grammar
from geneticengine.random.sources import RandomSource
from geneticengine.grammar.metahandlers.base import MetaHandlerGenerator, SynthesisException

T = TypeVar("T")


class VarRange(MetaHandlerGenerator):
    """VarRange([a, b, c]) represents the alternative between a, b, and c.

    The list of options can be dynamically altered before the grammar
    extraction with something like Var.__init__.__annotations__["name"]
    = Annotated[str, VarRange([d, e, f])]. The option list must not be
    empty.
    """

    def __init__(self, options: list[T]):
        if not options:
            raise SynthesisException(
                f"The VarRange metahandler requires a non-empty set of options. Options found: {options}",
            )
        self.options = options

    def validate(self, v) -> bool:
        return v in self.options

    def generate(
        self,
        random: RandomSource,
        grammar: Grammar,
        base_type: type,
        rec: Callable[[type[T]], T],
        dependent_values: dict[str, Any],
        parent_values: list[dict[str, Any]],
    ):
        return random.choice(self.options)

    def __repr__(self):
        return str(self.options)

    def __class_getitem__(cls, args):
        return VarRange(*args)

    def iterate(
        self,
        base_type: type,
        combine_lists: Callable[[list[type]], Generator[Any, Any, Any]],
        rec: Any,
        dependent_values: dict[str, Any],
    ):
        yield from self.options
