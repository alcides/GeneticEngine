from abc import ABC
from dataclasses import dataclass
from typing import Annotated

import numpy as np

from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.metahandlers.floats import FloatRange
from geneticengine.grammar.metahandlers.ints import IntervalRange
from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.grammar.metahandlers.lists import ListSizeBetween
from geneticengine.grammar.metahandlers.strings import WeightedStringHandler
from geneticengine.grammar.metahandlers.vars import VarRange


class Root(ABC):
    pass


@dataclass
class K(Root):
    b: Root
    i: int
    k: "K"


@dataclass
class Recursive(Root):
    left: K
    right: K


@dataclass
class IntRangeM(Root):
    x: Annotated[int, IntRange[9, 10]]


@dataclass
class IntervalRangeM(Root):
    x: Annotated[
        tuple[int, int],
        IntervalRange(
            minimum_length=5,
            maximum_length=10,
            maximum_top_limit=100,
        ),
    ]


@dataclass
class FloatRangeM(Root):
    x: Annotated[float, FloatRange[9.0, 10.0]]


@dataclass
class VarRangeM(Root):
    x: Annotated[str, VarRange(["x", "y", "z"])]


@dataclass
class ListRangeM(Root):
    x: Annotated[list[int], ListSizeBetween[3, 4]]


@dataclass
class WeightedStringHandlerM(Root):
    x: Annotated[
        str,
        WeightedStringHandler(
            np.array(
                [
                    [-0.01, 0.0425531914894, 0.01, 0.937446808511],
                    [0.97, 0.01, 0.01, 0.01],
                    [0.0212765957447, 0.01, 0.958723404255, 0.01],
                    [
                        0.106382978723,
                        0.0212765957447,
                        0.787234042553,
                        0.0851063829787,
                    ],
                    [0.533191489362, 0.01, 0.01, 0.446808510638],
                ],
            ),
            ["A", "C", "G", "T"],
        ),
    ]


grammar = extract_grammar(
    [Recursive, IntRangeM, IntervalRangeM, FloatRangeM, VarRangeM, ListRangeM, WeightedStringHandlerM],
    Root,
)
