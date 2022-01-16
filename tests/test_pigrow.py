from abc import ABC
from dataclasses import dataclass
from typing import Annotated, List
from geneticengine.core.random.sources import RandomSource

from geneticengine.core.grammar import Grammar, extract_grammar

from geneticengine.core.representations.treebased import random_node


class Root(ABC):
    pass


class A(Root):
    x: Root
    y: int
    z: Root


@dataclass
class Concrete(Root):
    x: int


@dataclass
class Middle(Root):
    x: Root


@dataclass
class ConcreteList(Root):
    xs: List[int]


class TestPIGrow(object):
    def test_root(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([Concrete], Root)
        x = random_node(r, g, 2, Root)
        assert isinstance(x, Concrete)
        assert isinstance(x, Root)

    def test_leaf(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([Concrete], Concrete)
        x = random_node(r, g, 2, Concrete)
        assert isinstance(x, Concrete)
        assert isinstance(x, Root)

    def test_leaf2(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([Concrete], Root)
        x = random_node(r, g, 2, Concrete)
        assert isinstance(x, Concrete)
        assert isinstance(x, Root)

    def test_list(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([ConcreteList], Root)
        x = random_node(r, g, 3, Root)
        assert isinstance(x, ConcreteList)
        assert isinstance(x.xs, list)
        assert isinstance(x, Root)

    def test_middle_has_right_distance_to_term(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([Concrete, Middle], Root)
        x = random_node(r, g, 10, Root)
        assert x.distance_to_term == 10
        assert isinstance(x, Root)
