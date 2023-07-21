from __future__ import annotations
import copy
import string

from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.tree.initializations import pi_grow_method
from geneticengine.metahandlers.base import MetaHandlerGenerator


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

    def generate(
        self,
        r: Source,
        g: Grammar,
        rec,
        new_symbol,
        depth: int,
        base_type,
        ctx: dict[str, str],
    ):
        size = r.randint(self.min, self.max, "str")
        s = "".join(r.choice(self.options) for _ in range(size))
        rec(s)

    def mutate(
        self,
        r: Source,
        g: Grammar,
        random_node,
        depth: int,
        base_type,
        current_node,
        method=pi_grow_method,
    ):
        mutation_method = r.randint(0, 2, "str")
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
        r: Source,
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

        size = r.randint(self.min, self.max, "str")
        midpoint = r.randint(1, size - 1)
        other = r.choice([getattr(x, arg) for x in options])
        return current_node[:midpoint] + other[midpoint:]

    def __class_getitem__(self, args):
        return StringSizeBetween(*args)

    def __repr__(self):
        return f"StringSizeBetween[{self.min}...{self.max}]"


class WeightedStringHandler(MetaHandlerGenerator):
    """This metahandler restricts the creation of string nodes so that the
    output complies with a given alphabet and a matrix of probabilities for
    each position.

    Each row on the matrix should reflect the probability of
    each character in that position. Thus, the number of cols
    in the input matrix should be the same as the number of
    characters in the alphabet.

    This refinement will return a string with a
    size == nrows in the matrix
    """

    def __init__(self, matrix, alphabet):
        self.probability_matrix = matrix
        self.alphabet = alphabet

        assert (
            len(self.alphabet) == self.probability_matrix.shape[1]
        ), "Cols in probability matrix must have the same size as the alphabet provided"

    def generate(
        self,
        r: Source,
        g: Grammar,
        rec,
        newsymbol,
        depth: int,
        base_type,
        context: dict[str, str],
    ):
        out = ""
        for row in self.probability_matrix:
            out += r.choice_weighted(self.alphabet, row, str(base_type))
        rec(out)

    def __repr__(self):
        return f"str[aphabet={self.alphabet}, size={self.probability_matrix.shape[0]}"

    def __class_getitem__(self, args):
        return WeightedStringHandler(*args)
