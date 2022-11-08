from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Annotated

from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.grammar import Grammar
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.random.sources import RandomSource
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


@dataclass
class ConcreteList(Root):
    xs: list[int]


class TestRamped:
    def test_ramped_half_and_half(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([Concrete, Middle], Root, False)
        problem=SingleObjectiveProblem(
            minimize=False,
            fitness_function=lambda x: x,
            target_fitness=None,
        )

        pop_size = 21
        max_init_d = 3
        for i in range(6):
            for j in range(5):
                max_id = max_init_d + j
                pop_s = pop_size + i
                gp = GP(
                    grammar = g,
                    problem=problem,
                    max_init_depth=max_id,
                    population_size=pop_s,
                )
                pop = gp.init_population(ramped_half_and_half=True)
                depths = list(map(lambda x: x.genotype.gengy_distance_to_term, pop))
                max_d_half = int(pop_s / 2)
                assert depths.count(max_id) == max_d_half + int((pop_s - max_d_half) / max_id)

a = TestRamped
a.test_ramped_half_and_half(a)
