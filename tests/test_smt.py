from abc import ABC
from dataclasses import dataclass
from typing import Annotated, List, TypeVar

from geneticengine.core.random.sources import RandomSource
from geneticengine.core.grammar import Grammar, extract_grammar
from geneticengine.core.representations.treebased import random_node
from geneticengine.metahandlers.smt.base import SMT


class Root(ABC):
    pass


@dataclass
class IntC(Root):
    x: Annotated[int, SMT("9 <= x && x <= 10")]


@dataclass
class BoolC(Root):
    x: Annotated[bool, SMT("x")]


@dataclass
class FloatC(Root):
    x: Annotated[float, SMT("x > 0.3")]


T = TypeVar("T")


class TestMetaHandler(object):
    def skeleton(self, t):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([t], Root)
        n = random_node(r, g, 3, Root)
        assert isinstance(n, Root)
        return n

    def test_int(self):
        n = self.skeleton(IntC)
        assert isinstance(n, IntC)
        assert 9 <= n.x <= 10

    def test_bool(self):
        n = self.skeleton(BoolC)
        assert isinstance(n, BoolC)
        assert n.x == True

    def test_real(self):
        n = self.skeleton(FloatC)
        assert isinstance(n, FloatC)
        assert n.x > 0.3
