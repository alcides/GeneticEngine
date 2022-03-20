from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import Type

from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import Source
from geneticengine.metahandlers.base import MetaHandlerGenerator


class WeightedStringHandler(MetaHandlerGenerator):
    """
    This metahandler restricts the creation of string nodes
    so that the output complies with a given alphabet and a
    matrix of probabilities for each position.

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
            out += r.choice_weighted(self.alphabet, row)
        rec(out)

    def __repr__(self):
        return f"str[aphabet={self.alphabet}, size={self.probability_matrix.shape[0]}"

    def __class_getitem__(self, args):
        return WeightedStringHandler(*args)
