from abc import ABC
from dataclasses import dataclass
from typing import Annotated, List

from geneticengine.core.decorators import abstract
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.grammar import Grammar, extract_grammar
from geneticengine.core.representations.tree.position_independent_grow import Future
from geneticengine.core.representations.tree.treebased import random_node
from geneticengine.core.utils import get_arguments
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.lists import ListSizeBetween


class Root(ABC):
    pass


@dataclass
class Leaf(Root):
    pass


@dataclass
class A(Root):
    y: List[Annotated[int, IntRange(7, 9)]]
    z: Annotated[List[Root], ListSizeBetween(2, 3)]


@dataclass
class Concrete(Root):
    x: int


@dataclass
class Middle(Root):
    x: Root


@dataclass
class ConcreteList(Root):
    xs: List[int]


def contains_future(t):
    if isinstance(t, Future):
        return True
    elif isinstance(t, list):
        for el in t:
            if contains_future(el):
                return True
    else:
        for (argn, argt) in get_arguments(t):
            if contains_future(getattr(t, argn)):
                return True
    return False


class TestPIGrow(object):
    def test_root(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([Concrete], Root)
        x = random_node(r, g, 2, Root)
        assert isinstance(x, Concrete)
        assert isinstance(x, Root)

    def test_leaf(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([Leaf], Concrete)
        x = random_node(r, g, 2, Leaf)
        assert isinstance(x, Leaf)
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

    def test_no_futures(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([Leaf, A], Root)
        x = random_node(r, g, 10, Root)
        assert not contains_future(x)
