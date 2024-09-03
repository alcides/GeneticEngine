from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Annotated

from geneticengine.grammar.grammar import extract_grammar
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.representations.tree.treebased import random_node
from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.grammar.metahandlers.lists import ListSizeBetween


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
        r = NativeRandomSource(seed=1)
        g = extract_grammar([Concrete], Root)
        decider = MaxDepthDecider(r, g, 4)
        x = random_node(r, g, Concrete, decider=decider)
        assert isinstance(x, Concrete)
        assert isinstance(x, Root)

    def test_leaf(self):
        r = NativeRandomSource(seed=1)
        g = extract_grammar([Leaf], Root)
        decider = MaxDepthDecider(r, g, 4)
        x = random_node(r, g, Leaf, decider=decider)
        assert isinstance(x, Leaf)
        assert isinstance(x, Root)

    def test_leaf2(self):
        r = NativeRandomSource(seed=1)
        g = extract_grammar([Concrete], Root)
        decider = MaxDepthDecider(r, g, 4)
        x = random_node(r, g, Concrete, decider=decider)
        assert isinstance(x, Concrete)
        assert isinstance(x, Root)

    def test_concrete_list(self):
        r = NativeRandomSource(seed=1)
        g = extract_grammar([ConcreteList], Root)
        decider = MaxDepthDecider(r, g, 6)
        x = random_node(r, g, Root, decider=decider)
        assert isinstance(x, ConcreteList)
        assert isinstance(x.xs, list)
        assert isinstance(x, Root)

    def test_concrete_annotated_list(self):
        r = NativeRandomSource(seed=1)
        g = extract_grammar([ConcreteAnnotatedList], Root)
        decider = MaxDepthDecider(r, g, 6)
        x = random_node(r, g, Root, decider=decider)
        assert isinstance(x, ConcreteAnnotatedList)
        assert isinstance(x.xs, list)
        assert isinstance(x, Root)

    def test_middle_list(self):
        r = NativeRandomSource(seed=1)
        g = extract_grammar([MiddleList, Concrete], Root)
        decider = MaxDepthDecider(r, g, 6)
        x = random_node(r, g, Root, decider=decider)
        assert (isinstance(x, MiddleList) and isinstance(x.z, list)) or isinstance(x, Concrete)
        assert isinstance(x, Root)

    def test_middle_has_right_distance_to_term(self) -> None:
        @dataclass
        class RootHolder:  # a holder is needed to know the true height, because choosing consumes height
            root: Root

        r = NativeRandomSource(seed=1)
        g = extract_grammar([Concrete, Middle, Root], RootHolder)
        decider = MaxDepthDecider(r, g, 20)
        x = random_node(r, g, RootHolder, decider=decider)
        assert isinstance(x, RootHolder)
