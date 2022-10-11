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


class TestDepthing:
    def test_normal_depthing(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([Concrete], Root, False)
        x = random_node(r, g, 4, Root, method=PI_Grow)
        g: Grammar = extract_grammar([Concrete, Middle], Root, False)
        y = random_node(r, g, 4, Root, method=PI_Grow)
        g: Grammar = extract_grammar([Concrete, Middle, MiddleList], Root, False)
        z = random_node(r, g, 4, Root, method=PI_Grow)
        assert x.gengy_distance_to_term == 1
        assert x.gengy_nodes == 1
        assert x.gengy_weighted_nodes == 1
        assert y.gengy_distance_to_term == 4
        assert y.gengy_nodes == 4
        assert y.gengy_weighted_nodes == 10
        assert z.gengy_distance_to_term == 4
        assert z.gengy_nodes == 7
        assert z.gengy_weighted_nodes == 16

    def test_expansion_depthing(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([Concrete], Root, True)
        x = random_node(r, g, 4, Root, method=PI_Grow)
        g: Grammar = extract_grammar([Concrete, Middle], Root, True)
        y = random_node(r, g, 8, Root, method=PI_Grow)
        g: Grammar = extract_grammar([Concrete, Middle, MiddleList], Root, True)
        z = random_node(r, g, 8, Root, method=PI_Grow)
        assert x.gengy_distance_to_term == 2
        assert x.gengy_nodes == 2
        assert x.gengy_weighted_nodes == 3
        assert y.gengy_distance_to_term == 6
        assert y.gengy_nodes == 6
        assert y.gengy_weighted_nodes == 13
        assert z.gengy_distance_to_term == 6
        assert z.gengy_nodes == 14
        assert z.gengy_weighted_nodes == 26
