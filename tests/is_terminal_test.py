from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Annotated

from geneticengine.core.grammar import extract_grammar
from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.tree.treebased import PI_Grow
from geneticengine.core.representations.tree.treebased import random_node
from geneticengine.core.representations.tree.utils import GengyList
from geneticengine.core.representations.tree.utils import relabel_nodes
from geneticengine.core.utils import has_arguments
from geneticengine.core.utils import is_terminal
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.lists import ListSizeBetween


class Root(ABC):
    pass


@dataclass
class Leaf(Root):
    pass


@dataclass
class A(Root):
    # y: GengyList[Annotated[int, IntRange(7, 9)]]
    z: Annotated[GengyList[Root], ListSizeBetween(2, 3)]


@dataclass
class Concrete(Root):
    x: int


@dataclass
class Middle(Root):
    x: Root


@dataclass
class ConcreteList(Root):
    xs: GengyList[int]


class TestIsTerminal:
    def test_terminals(self):
        g: Grammar = extract_grammar([Leaf], Root)
        assert is_terminal(Leaf, g.non_terminals)

    def test_non_terminals(self):
        g: Grammar = extract_grammar([Concrete, Middle, ConcreteList, A], Root)
        assert not is_terminal(Concrete, g.non_terminals)
        assert not is_terminal(Middle, g.non_terminals)
        assert not is_terminal(ConcreteList, g.non_terminals)
        assert not is_terminal(A, g.non_terminals)
        assert not is_terminal(Root, g.non_terminals)
