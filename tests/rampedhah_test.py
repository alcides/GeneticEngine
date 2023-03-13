from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Annotated

from geneticengine.core.grammar import extract_grammar
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.tree.operators import (
    RampedHalfAndHalfInitializer,
)
from geneticengine.core.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.metahandlers.lists import ListSizeBetween


class Root(ABC):
    pass


@dataclass
class Leaf(Root):
    pass


@dataclass
class MiddleList(Root):
    z: Annotated[list[Root], ListSizeBetween(2, 3)]


@dataclass
class Concrete(Root):
    x: int


@dataclass
class Middle(Root):
    x: Root


class TestRamped:
    def test_ramped_half_and_half(self):
        r = RandomSource(seed=1)
        g = extract_grammar([Concrete, Middle], Root, False)
        problem = SingleObjectiveProblem(
            minimize=False,
            fitness_function=lambda x: x,
        )

        max_depth = 10
        pop = RampedHalfAndHalfInitializer().initialize(
            problem,
            TreeBasedRepresentation(g, max_depth=max_depth),
            r,
            2,
        )
        print([i.genotype for i in pop])
        depths = list(map(lambda x: x.genotype.gengy_distance_to_term, pop))
        assert depths[0] != depths[-1]
