from abc import ABC
from dataclasses import dataclass
from geneticengine.algorithms.gp.gp import GP

from geneticengine.core.random.sources import RandomSource
from geneticengine.core.grammar import Grammar, extract_grammar
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
class UnderTest(object):
    a: Leaf
    b: Root


class TestGrammar(object):
    def test_safety(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([Leaf, OtherLeaf, UnderTest, Root], UnderTest)
        g = GP(
            grammar=g,
            evaluation_function=lambda x: 1,
            population_size=10,
            number_of_generations=100,
        )
        (_, _, x) = g.evolve()
        assert isinstance(x.a, Leaf)
        assert isinstance(x, UnderTest)
