from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Annotated

import pytest
from geneticengine.algorithms.gp.gp import GP

from geneticengine.core.grammar import extract_grammar
from geneticengine.core.grammar import Grammar
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.tree2.treebased2 import TreeBased2Representation
from geneticengine.metahandlers.floats import FloatRange
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.lists import ListSizeBetween
from geneticengine.metahandlers.vars import VarRange


class Root(ABC):
    pass


@dataclass
class IntRangeM(Root):
    x: Annotated[int, IntRange[9, 10]]


@dataclass
class Branch(Root):
    a: Root
    b: Root


@dataclass
class Concrete(Root):
    a: int


@dataclass
class FloatRangeM(Root):
    x: Annotated[float, FloatRange[9.0, 10.0]]


@dataclass
class VarRangeM(Root):
    x: Annotated[str, VarRange(["x", "y", "z"])]


@dataclass
class ListRangeM(Root):
    x: Annotated[list[int], ListSizeBetween[3, 4]]


class TestRepresentation:
    @pytest.mark.parametrize(
        "representation_class",
        [
            # TreeBasedRepresentation,
            # GrammaticalEvolutionRepresentation,
            # DynamicStructuredGrammaticalEvolutionRepresentation,
            # StackBasedGGGPRepresentation,
            # SMTTreeBasedRepresentation,
            TreeBased2Representation,
        ],
    )
    def test_rep(self, representation_class) -> None:
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([IntRangeM, ListRangeM, FloatRangeM, Branch, Concrete], Root)
        max_depth = 3

        repr = representation_class(g, max_depth)

        def fitness_function(x: Root) -> float:
            return 0.5

        gp = GP(representation=repr, problem=SingleObjectiveProblem(fitness_function=fitness_function), random_source=r)
        gp.evolve()
