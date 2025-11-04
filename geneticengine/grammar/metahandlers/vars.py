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
    """Like VarRange but allows specifying per-option selection probabilities.

    Usage: Annotated[str, VarRangeWithProbabilities(options, weights)]
    """

    def __init__(self, options: list[T], weights: Sequence[float]):
        if not options:
            raise SynthesisException(
                f"The VarRangeWithProbabilities metahandler requires a non-empty set of options. Options found: {options}",
            )
        if not weights or len(weights) != len(options):
            raise SynthesisException(
                "The weights list must be the same length as options and non-empty.",
            )
        self.options = options
        self.weights = list(weights)

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
        total = sum(self.weights)
        if total <= 0:
            return random.choice(self.options)
        normalized = [w / total if w >= 0 else 0.0 for w in self.weights]
        return random.choice_weighted(self.options, normalized)

    def __repr__(self):
        return f"{self.options} (weighted)"

    def __class_getitem__(cls, args):
        return VarRangeWithProbabilities(*args)

    def iterate(
        self,
        base_type: type,
        combine_lists: Callable[[list[type]], Generator[Any, Any, Any]],
        rec: Any,
        dependent_values: dict[str, Any],
    ):
        yield from self.options
