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
class A(Root):
    # y: list[int]
    z: Annotated[GengyList[Root], ListSizeBetween(2, 3)]


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
        g: Grammar = extract_grammar([Concrete, Middle, A], Root, False)
        z = random_node(r, g, 4, Root, method=PI_Grow)
        assert x.gengy_distance_to_term == 1
        assert x.gengy_nodes == 1
        assert y.gengy_distance_to_term == 4
        assert y.gengy_nodes == 4
        # assert z.gengy_distance_to_term == 4
        # assert z.gengy_nodes == 7
        # import IPython as ip
        # ip.embed()

    def test_expansion_depthing(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([Concrete], Root, True)
        x = random_node(r, g, 4, Root, method=PI_Grow)
        g: Grammar = extract_grammar([Concrete, Middle], Root, True)
        y = random_node(r, g, 8, Root, method=PI_Grow)
        assert x.gengy_distance_to_term == 2
        assert x.gengy_nodes == 2
        assert y.gengy_distance_to_term == 6
        assert y.gengy_nodes == 6
        # import IPython as ip
        # ip.embed()


# a = TestDepthing
# a.test_normal_depthing(a)
# a.test_expansion_depthing(a)
