from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Annotated
from typing import List

from geneticengine.core.grammar import extract_grammar
from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.tree.treebased import PI_Grow
from geneticengine.core.representations.tree.treebased import random_node
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.lists import ListSizeBetween


class Root(ABC):
    pass


@dataclass
class Leaf(Root):
    pass


@dataclass
class A(Root):
    y: list[Annotated[int, IntRange(7, 9)]]
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


class TestPIGrow:
    def test_root(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar(Root, globals(), [Concrete])
        x = random_node(r, g, 4, Root, method=PI_Grow)
        assert isinstance(x, Concrete)
        assert isinstance(x, Root)

    def test_leaf(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar(Concrete, globals(), [Leaf])
        x = random_node(r, g, 4, Leaf, method=PI_Grow)
        assert isinstance(x, Leaf)
        assert isinstance(x, Root)

    def test_leaf2(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar(Root, globals(), [Concrete])
        x = random_node(r, g, 4, Concrete, method=PI_Grow)
        assert isinstance(x, Concrete)
        assert isinstance(x, Root)

    def test_list(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar(Root, globals(), [ConcreteList])
        x = random_node(r, g, 6, Root, method=PI_Grow)
        assert isinstance(x, ConcreteList)
        assert isinstance(x.xs, list)
        assert isinstance(x, Root)

    def test_middle_has_right_distance_to_term(self):
        @dataclass
        class RootHolder:  # a holder is needed to know the true height, because choosing consumes height
            root: Root

        r = RandomSource(seed=1)
        g: Grammar = extract_grammar(RootHolder, globals(), [Concrete, Middle, Root])
        x = random_node(r, g, 20, RootHolder, method=PI_Grow)
        assert x.gengy_distance_to_term == 20
        assert isinstance(x, RootHolder)
