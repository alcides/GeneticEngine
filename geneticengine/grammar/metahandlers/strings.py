from __future__ import annotations
import copy
import itertools
import string
from typing import Any, Callable, Generator, TypeVar

from geneticengine.grammar.grammar import Grammar
from geneticengine.random.sources import RandomSource
from geneticengine.grammar.metahandlers.base import MetaHandlerGenerator

T = TypeVar("T")


class StringSizeBetween(MetaHandlerGenerator):
    """StringSizeBetween(a,b) restricts strings to be of length between a and b
    and implements a special list mutation.

    The range can be dynamically altered before the grammar extraction
        X.__annotations__["y"] = Annotated[List[Type], ListSizeBetween(c,d)].
    The special string mutation entails three different alterations to the list in question: deletion of a random element;
        addition of a random element; and replacement of a random element.
    """

    def __init__(self, min, max, options=string.ascii_letters + string.digits):
        self.min = min
        self.max = max
        self.options = list(options)

    def validate(self, v) -> bool:
        return self.min <= len(v) <= self.max and all(x in self.options for x in v)

    def generate(
        self,
        random: RandomSource,
        grammar: Grammar,
        base_type: type,
        rec: Callable[[type[T]], T],
        dependent_values: dict[str, Any],
        parent_values: list[dict[str, Any]],
    ) -> Any:
        size = random.randint(self.min, self.max)
        s = "".join(random.choice(self.options) for _ in range(size))
        return s

    def mutate(
        self,
        r: RandomSource,
        g: Grammar,
        random_node,
        depth: int,
        base_type,
        current_node,
    ):
        mutation_method = r.randint(0, 2)
        current_str = copy.copy(current_node)
        if (mutation_method == 0) and (len(current_node) > self.min):  # del
            element_to_be_deleted = r.randint(0, len(current_node) - 1)
            return current_str[:element_to_be_deleted] + current_str[element_to_be_deleted + 1 :]
        elif (mutation_method == 1) and (len(current_node) < self.max):  # add
            s = r.choice(self.options)
            element_to_be_added = r.randint(0, len(current_node) - 1)
            return current_str[:element_to_be_added] + s + current_str[element_to_be_added:]
        elif len(current_node) > 0:  # replace
            element_to_be_replaced = r.randint(0, len(current_node) - 1)
            s = r.choice(self.options)
            return current_str[:element_to_be_replaced] + s + current_str[element_to_be_replaced + 1 :]
        else:
            return current_str

    def crossover(
        self,
        r: RandomSource,
        g: Grammar,
        options,
        arg,
        list_type,
        current_node,
    ):
        if not options or (len(current_node) < 2):
            return current_node

        #  [3,6]
        #  4 from the first str    |          2 from the first str
        #  bound = [0,2]           |          bound = [1,4]

        size = r.randint(self.min, self.max)
        midpoint = r.randint(1, size - 1)
        other = r.choice([getattr(x, arg) for x in options])
        return current_node[:midpoint] + other[midpoint:]

    def __class_getitem__(cls, args):
        return StringSizeBetween(*args)

    def __repr__(self):
        return f"StringSizeBetween[{self.min}...{self.max}]"

    def iterate(
        self,
        base_type: type,
        combine_lists: Callable[[list[type]], Generator[Any, Any, Any]],
        rec: Any,
        dependent_values: dict[str, Any],
    ):
        def generate_letter():
            yield from self.options

        for length in range(self.min, self.max + 1):
            yield from itertools.product(*(generate_letter() for _ in range(length)))


class WeightedStringHandler(MetaHandlerGenerator):
    """This metahandler restricts the creation of string nodes so that the
    output complies with a given alphabet and a matrix of probabilities for
    each position.

    Each row on the matrix should reflect the probability of each
    character in that position. Thus, the number of cols in the input
    matrix should be the same as the number of characters in the
    alphabet.

    This refinement will return a string with a size == nrows in the
    matrix
    """

    def __init__(self, matrix, alphabet):
        self.probability_matrix = matrix
        self.alphabet = alphabet

        assert (
            len(self.alphabet) == self.probability_matrix.shape[1]
        ), "Cols in probability matrix must have the same size as the alphabet provided"

    def validate(self, v) -> bool:
        return len(v) == self.probability_matrix.shape[0] and all(x in self.alphabet for x in v)

    def generate(
        self,
        random: RandomSource,
        grammar: Grammar,
        base_type: type,
        rec: Callable[[type[T]], T],
        dependent_values: dict[str, Any],
        parent_values: list[dict[str, Any]],
    ) -> str:
        out = ""
        for row in self.probability_matrix:
            out += random.choice_weighted(self.alphabet, row)
        return out

    def __repr__(self):
        return f"str[aphabet={self.alphabet}, size={self.probability_matrix.shape[0]}"

    def __class_getitem__(cls, args):
        return WeightedStringHandler(*args)

    def iterate(
        self,
        base_type: type,
        combine_lists: Callable[[list[type]], Generator[Any, Any, Any]],
    ):
        def generate_letter():
            yield from self.options

        yield from itertools.product(*(generate_letter() for _ in range(self.probability_matrix.shape[0])))
