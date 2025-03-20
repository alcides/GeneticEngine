from __future__ import annotations

from typing import Any, Callable, Generator, TypeVar

from geneticengine.grammar.grammar import Grammar
from geneticengine.random.sources import RandomSource
from geneticengine.grammar.metahandlers.base import MetaHandlerGenerator

min = TypeVar("min", covariant=True)
max = TypeVar("max", covariant=True)

T = TypeVar("T")


class IntRange(MetaHandlerGenerator):
    """IntRange(a,b) restricts ints to be between a and b.

    The range can be dynamically altered before the grammar extraction
    Int.__init__.__annotations__["value"] = Annotated[int,
    IntRange(c,d)]
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def generate(
        self,
        random: RandomSource,
        grammar: Grammar,
        base_type: type,
        rec: Callable[[type[T]], T],
        dependent_values: dict[str, Any],
        parent_values: list[dict[str, Any]],
    ):
        return random.randint(self.min, self.max)

    def validate(self, v) -> bool:
        return self.min <= v <= self.max

    def iterate(
        self,
        base_type: type,
        combine_lists: Callable[[list[type]], Generator[Any, Any, Any]],
        rec: Any,
        dependent_values: dict[str, Any],
    ):
        for i in range(self.min, self.max + 1):
            yield i

    def __class_getitem__(cls, args):
        return IntRange(*args)

    def __repr__(self):
        return f"[{self.min},..,{self.max}]"


# TODO: deprecate
class IntervalRange(MetaHandlerGenerator):
    """This metahandler restricts the creation of ranges between two integers
    by forcing a minimum and maximum range size, as well as a top limit that
    the range can reach.

    This is useful in genomics to generate random windows of variable
    size to scan an input sequence
    """

    def __init__(
        self,
        minimum_length: int,
        maximum_length: int,
        maximum_top_limit: int,
    ):
        """
        :param int minimum_length: Minimum length possible when randomly generating the range
        :param int maximum_length: Maximum length possible when randomly generating the range
        :param int maximum_top_limit: Maximum value the range can reach
        """
        assert maximum_length > minimum_length
        assert maximum_length < maximum_top_limit

        self.minimum_length = minimum_length
        self.maximum_length = maximum_length
        self.maximum_top_limit = maximum_top_limit

    def __class_getitem__(cls, args):
        return IntervalRange(*args)

    def generate(
        self,
        random: RandomSource,
        grammar: Grammar,
        base_type: type,
        rec: Callable[[type[T]], T],
        dependent_values: dict[str, Any],
        parent_values: list[dict[str, Any]],
    ):

        range_length = random.randint(self.minimum_length, self.maximum_length)
        start_position = random.randint(0, self.maximum_top_limit - range_length)
        return (start_position, start_position + range_length)

    def validate(self, v) -> bool:
        length = v[1] - v[0]
        return self.minimum_length < length <= self.maximum_length and v[1] < self.maximum_top_limit

    def iterate(
        self,
        base_type: type,
        combine_lists: Callable[[list[type]], Generator[Any, Any, Any]],
    ):
        for i in range(self.maximum_length, self.maximum_top_limit + 1):
            yield self.minimum_length + i
