from __future__ import annotations

from typing import Any, Callable, Generator, TypeVar, Sequence

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


class VarRangeWithProbabilities(MetaHandlerGenerator):
    """VarRangeWithProbabilities([a, b, c], [pa, pb, pc]) represents the
    alternative between a, b, and c with the given probabilities.

    The options list must not be empty and the probabilities list must have
    the same length as options. Probabilities can be any non-negative numbers
    and are interpreted as weights.
    """

    def __init__(self, options: list[T], probabilities: list[float]):
        if not options:
            raise SynthesisException(
                f"The VarRangeWithProbabilities metahandler requires a non-empty set of options. Options found: {options}",
            )
        if len(options) != len(probabilities):
            raise SynthesisException(
                "Options and probabilities must have the same length.",
            )
        if any(p < 0 for p in probabilities):
            raise SynthesisException(
                "Probabilities must be non-negative.",
            )
        self.options = options
        self.probabilities = probabilities

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
        return random.choice_weighted(self.options, self.probabilities)

    def __repr__(self):
        return f"{self.options} with probabilities {self.probabilities}"

    def __class_getitem__(cls, args):
        return VarRangeWithProbabilities(*args)

    def iterate(
        self,
        base_type: type,
        combine_lists: Callable[[list[type]], Generator[Any, Any, Any]],
        rec: Any,
        dependent_values: dict[str, Any],
    ):
        indexed = list(enumerate(self.options))
        # Sort by probability descending; stable by index to preserve order on ties
        order = sorted(indexed, key=lambda t: (-self.probabilities[t[0]], t[0]))
        for _, opt in order:
            yield opt
