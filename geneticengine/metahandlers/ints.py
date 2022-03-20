from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import Type
from typing import TypeVar

from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import Source
from geneticengine.metahandlers.base import MetaHandlerGenerator

min = TypeVar("min", covariant=True)
max = TypeVar("max", covariant=True)


class IntRange(MetaHandlerGenerator):
    """
    IntRange(a,b) restricts ints to be between a and b.
    The range can be dynamically altered before the grammar extraction (Int.__init__.__annotations__["value"] = Annotated[int, IntRange(c,d)].
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
        context: dict[str, str],
    ):
        rec(r.randint(self.min, self.max))

    def __class_getitem__(self, args):
        return IntRange(*args)

    def __repr__(self):
        return f"[{self.min}...{self.max}]"


class IntervalRange(MetaHandlerGenerator):
    """
    This metahandler restricts the creation of ranges
    between two integers by forcing a minimum and maximum
    range size, as well as a top limit that the range
    can reach

    This is useful in genomics to generate random windows
    of variable size to scan an input sequence
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

    def __class_getitem__(self, args):
        return IntervalRange(*args)

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

        range_length = r.randint(self.minimum_length, self.maximum_length)
        start_position = r.randint(0, self.maximum_top_limit - range_length)
        rec((start_position, start_position + range_length))
