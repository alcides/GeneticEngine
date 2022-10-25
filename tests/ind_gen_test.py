from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Annotated
from typing import List

from geneticengine.core.grammar import extract_grammar
from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.tree.treebased import Full
from geneticengine.core.representations.tree.treebased import PI_Grow
from geneticengine.core.representations.tree.treebased import Ramped_HalfAndHalf
from geneticengine.core.representations.tree.treebased import random_node
from geneticengine.core.representations.tree.utils import GengyList
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.lists import ListSizeBetween


class Root(ABC):
    pass


@dataclass
class Leaf(Root):
    pass


@dataclass
class MiddleList(Root):
    xs: list[Annotated[int, IntRange(0, 9)]]
    z: Annotated[list[Root], ListSizeBetween(2, 3)]


@dataclass
class Concrete(Root):
    x: int


@dataclass
class Middle(Root):
    x: Root


@dataclass
class MiddleDouble(Root):
    x: Root
    y: Root


@dataclass
class ConcreteList(Root):
    xs: list[int]


@dataclass
class ConcreteAnnotatedList(Root):
    xs: list[Annotated[int, IntRange(0, 9)]]


class TestPIGrow:
    def test_root(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([Concrete], Root)
        x = random_node(r, g, 4, Root, method=PI_Grow)
        assert isinstance(x, Concrete)
        assert isinstance(x, Root)

    def test_leaf(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([Leaf], Root)
        x = random_node(r, g, 4, Leaf, method=PI_Grow)
        assert isinstance(x, Leaf)
        assert isinstance(x, Root)

    def test_leaf2(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([Concrete], Root)
        x = random_node(r, g, 4, Concrete, method=PI_Grow)
        assert isinstance(x, Concrete)
        assert isinstance(x, Root)

    def test_concrete_list(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([ConcreteList], Root)
        x = random_node(r, g, 6, Root, method=PI_Grow)
        assert isinstance(x, ConcreteList)
        assert isinstance(x.xs, list)
        assert isinstance(x, Root)

    def test_concrete_annotated_list(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([ConcreteAnnotatedList], Root)
        x = random_node(r, g, 6, Root, method=PI_Grow)
        assert isinstance(x, ConcreteAnnotatedList)
        assert isinstance(x.xs, list)
        assert isinstance(x, Root)

    def test_middle_list(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([MiddleList, Concrete], Root)
        x = random_node(r, g, 6, Root, method=PI_Grow)
        assert isinstance(x, MiddleList)
        assert isinstance(x.z, list)
        assert isinstance(x, Root)

    def test_middle_has_right_distance_to_term(self):
        @dataclass
        class RootHolder:  # a holder is needed to know the true height, because choosing consumes height
            root: Root

        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([Concrete, Middle, Root], RootHolder)
        x = random_node(r, g, 20, RootHolder, method=PI_Grow)
        assert x.gengy_distance_to_term == 20
        assert isinstance(x, RootHolder)


class TestFull:
    def test_root(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([Concrete], Root)
        x = random_node(r, g, 4, Root, method=Full)
        assert isinstance(x, Concrete)
        assert isinstance(x, Root)

    def test_middle_double(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([Concrete, MiddleDouble], Root)
        x = random_node(r, g, 4, Root, method=Full)
        assert x.gengy_nodes == 15
        y = random_node(r, g, 5, Root, method=Full)
        assert y.gengy_nodes == 31
        z = random_node(r, g, 6, Root, method=Full)
        assert z.gengy_nodes == 63
        x1 = random_node(r, g, 7, Root, method=Full)
        assert x1.gengy_nodes == 127
        import IPython as ip

        ip.embed()


class TestRamped:
    def test_root(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([Concrete], Root)
        x = random_node(r, g, 4, Root, method=Ramped_HalfAndHalf)
        assert isinstance(x, Concrete)
        assert isinstance(x, Root)

    def test_middle_double(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([Concrete, MiddleDouble], Root)
        individuals = []
        depths = []
        nodes = []
        for _ in range(100):
            x = random_node(r, g, 9, Root, method=Ramped_HalfAndHalf)
            individuals.append(x)
            depths.append(x.gengy_distance_to_term)
            nodes.append(x.gengy_nodes)

        assert max(depths) == 9
        assert depths.count(max(depths)) > 45 and depths.count(max(depths)) < 60
        assert max(nodes) == 511
        assert nodes.count(max(nodes)) > 45 and nodes.count(max(nodes)) < 55
