from abc import ABC
from dataclasses import dataclass
from typing import Annotated, List, Type

from geneticengine.core.decorators import abstract
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.grammar import Grammar, extract_grammar
from geneticengine.core.representations.tree.position_independent_grow import Future
from geneticengine.core.representations.tree.treebased import random_node
from geneticengine.core.utils import get_arguments
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.lists import ListSizeBetween


class Root(ABC):
    pass


@dataclass
class Leaf(Root):
    pass


@dataclass
class OtherLeaf(Root):
    pass


@dataclass
class UnderTest(object):
    a: Leaf
    b: Root


class TestGrammar(object):
    def test_safety(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([Leaf, OtherLeaf, UnderTest, Root], UnderTest)
        x = random_node(r, g, 2, UnderTest)
        assert isinstance(x.a, Leaf)
        assert isinstance(x, UnderTest)
