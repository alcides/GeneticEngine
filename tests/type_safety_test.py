from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from geneticengine.algorithms.gp.callback import Callback
from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.tree.treebased import random_node


class Root(ABC):
    pass


@dataclass
class Leaf(Root):
    pass


@dataclass
class OtherLeaf(Root):
    pass


@dataclass
class UnderTest(Root):
    a: Leaf
    b: Root


class TestCallBack(Callback):
    def process_iteration(self, generation: int, population, time: float):
        for ind in population:
            x = ind.genotype
            assert isinstance(x, UnderTest)
            assert isinstance(x.a, Leaf)


class TestGrammar:
    def test_safety(self):
        r = RandomSource(seed=123)
        g: Grammar = extract_grammar(UnderTest, globals())
        gp = GP(
            grammar=g,
            randomSource=lambda x: r,
            evaluation_function=lambda x: x.gengy_nodes,
            population_size=20,
            number_of_generations=10,
            probability_mutation=1,
            probability_crossover=1,
            max_depth=10,
            callbacks=[TestCallBack()],
        )
        (_, _, x) = gp.evolve()
        assert isinstance(x, UnderTest)
        assert isinstance(x.a, Leaf)
