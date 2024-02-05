from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Annotated

from geneticengine.grammar.grammar import extract_grammar
from geneticengine.solutions.tree import GengyList
from geneticengine.grammar.utils import is_terminal
from geneticengine.grammar.metahandlers.lists import ListSizeBetween


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
        g = extract_grammar([Leaf], Root)
        assert is_terminal(Leaf, g.non_terminals)

    def test_non_terminals(self):
        g = extract_grammar([Concrete, Middle, ConcreteList, A], Root)
        assert not is_terminal(Concrete, g.non_terminals)
        assert not is_terminal(Middle, g.non_terminals)
        assert not is_terminal(ConcreteList, g.non_terminals)
        assert not is_terminal(A, g.non_terminals)
        assert not is_terminal(Root, g.non_terminals)
