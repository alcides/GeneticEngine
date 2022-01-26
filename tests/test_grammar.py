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
class Rec(Root):
    n1: Root
    n2: Root
    n3: Root
    n4: Root
    n5: Root

@dataclass
class RecAlt(Root):
    l: Leaf

def contains_type(t, ty:Type):
    if isinstance(t, ty):
        return True
    elif isinstance(t, list):
        for el in t:
            if contains_type(el, ty):
                return True
    else:
        for (argn, argt) in get_arguments(t):
            if contains_type(getattr(t, argn), ty):
                return True
    return False


class TestGrammar(object):
    def test_root(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([Leaf], Root)
        x = random_node(r, g, 2, Root)
        assert isinstance(x, Leaf)
        assert isinstance(x, Root)
    
    def test_rec(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar([Leaf, Rec], Root)
        x = random_node(r, g, 2, Root)
        assert isinstance(x, Rec)
        assert isinstance(x, Root)

    def test_rec_alt(self):
        r = RandomSource(seed=2)
        g: Grammar = extract_grammar([Leaf, Rec, RecAlt], Root)
        x = random_node(r, g, max_depth=5, starting_symbol=Root, force_depth=5)
        assert contains_type(x, RecAlt)
        assert isinstance(x, Root)

