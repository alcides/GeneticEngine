from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Annotated

from geneticengine.core.grammar import extract_grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.tree.treebased import pi_grow_method
from geneticengine.core.representations.tree.treebased import random_node
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
    def test_normal_depthing(self) -> None:
        r = RandomSource(seed=1)
        g = extract_grammar([Concrete], Root, False)
        x = random_node(r, g, 4, Root, method=pi_grow_method)
        g = extract_grammar([Concrete, Middle], Root, False)
        y = random_node(r, g, 4, Root, method=pi_grow_method)
        g = extract_grammar([Concrete, Middle, MiddleList], Root, False)
        z = random_node(r, g, 4, Root, method=pi_grow_method)
        g = extract_grammar([ConcreteList, Middle], Root, False)
        a = random_node(r, g, 4, Root, method=pi_grow_method)
        assert x.gengy_distance_to_term == 1
        assert x.gengy_nodes == 1
        assert x.gengy_weighted_nodes == 1
        assert y.gengy_distance_to_term == 4
        assert y.gengy_nodes == 4
        assert y.gengy_weighted_nodes == 10
        assert z.gengy_distance_to_term == 4
        assert z.gengy_nodes == 4
        assert z.gengy_weighted_nodes == 10
        assert a.gengy_distance_to_term == 4
        assert a.gengy_nodes == 4
        assert a.gengy_weighted_nodes == 10

    def test_expansion_depthing(self) -> None:
        r = RandomSource(seed=1)
        g = extract_grammar([Concrete], Root, True)
        x = random_node(r, g, 4, Root, method=pi_grow_method)
        g = extract_grammar([Concrete, Middle], Root, True)
        y = random_node(r, g, 8, Root, method=pi_grow_method)
        g = extract_grammar([Concrete, Middle, MiddleList], Root, True)
        z = random_node(r, g, 8, Root, method=pi_grow_method)
        assert x.gengy_distance_to_term == 2
        assert x.gengy_nodes == 2
        assert x.gengy_weighted_nodes == 3
        assert y.gengy_distance_to_term == 6
        assert y.gengy_nodes == 6
        assert y.gengy_weighted_nodes == 13
        assert z.gengy_distance_to_term == 6
        assert z.gengy_nodes == 10
        assert z.gengy_weighted_nodes == 20
