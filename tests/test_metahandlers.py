from abc import ABC
from dataclasses import dataclass
from typing import Annotated, List
from geneticengine.core.random.sources import RandomSource

from geneticengine.core.grammar import Grammar, extract_grammar

from geneticengine.core.representations.tree.treebased import random_node

from geneticengine.metahandlers.ints import IntRange

from geneticengine.metahandlers.floats import FloatRange

from geneticengine.metahandlers.vars import VarRange

from geneticengine.metahandlers.lists import ListSizeBetween


class Root(ABC):
    pass


@dataclass
class IntRangeM(Root):
    x: Annotated[int, IntRange(9, 10)]


@dataclass
class FloatRangeM(Root):
    x: Annotated[float, FloatRange(9.0, 10.0)]


@dataclass
class VarRangeM(Root):
    x: Annotated[str, VarRange(["x", "y", "z"])]


@dataclass
class ListRangeM(Root):
    x: Annotated[List[int], ListSizeBetween(3, 4)]


class TestMetaHandler(object):
    def test_int(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([IntRangeM], Root)
        n = random_node(r, g, 3, Root)
        assert isinstance(n, IntRangeM)
        assert 9 <= n.x <= 10
        assert isinstance(n, Root)

    def test_float(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([FloatRangeM], Root)
        n = random_node(r, g, 3, Root)
        assert isinstance(n, FloatRangeM)
        assert 9.0 <= n.x <= 10.0
        assert isinstance(n, Root)

    def test_var(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([VarRangeM], Root)
        n = random_node(r, g, 3, Root)
        assert isinstance(n, VarRangeM)
        assert n.x in ["x", "y", "z"]
        assert isinstance(n, Root)

    def test_list(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([ListRangeM], Root)
        n = random_node(r, g, 4, Root)
        assert isinstance(n, ListRangeM)
        assert 3 <= len(n.x) <= 4
        assert isinstance(n, Root)
