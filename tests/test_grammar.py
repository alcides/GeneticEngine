from abc import ABC
from dataclasses import dataclass
from typing import Annotated, List, Type

from scipy import rand
from geneticengine.algorithms.gp.gp import GP
from geneticengine.algorithms.gp.individual import Individual

from geneticengine.core.decorators import abstract
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.grammar import Grammar, extract_grammar
from geneticengine.core.representations.tree.treebased import Grow, PI_Grow, random_node
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


@dataclass
class RecAlt(Root):
    l: Leaf


def contains_type(t, ty: Type):
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
        x = random_node(r, g, 10, Root, method=PI_Grow)
        #print(x) -- Leaf()
        assert isinstance(x, Rec)
        assert isinstance(x, Root)

    def test_rec_alt(self):
        r = RandomSource(seed=245)
        g: Grammar = extract_grammar([Leaf, Rec, RecAlt], Root)
        x = random_node(r, g, max_depth=15, starting_symbol=Root, method=PI_Grow)
        assert contains_type(x, RecAlt)
        assert isinstance(x, Root)


    def test_depth_increases(self):
        g: Grammar = extract_grammar([Leaf, Rec], Root)

        x = random_node(RandomSource(3), g, max_depth=2, starting_symbol=Root, method=PI_Grow)
        assert isinstance(x, Leaf)
        assert isinstance(x, Root)

        gp = GP(g, evaluation_function=lambda x: x.depth, randomSource=RandomSource, max_depth=5, seed=5)

        nx = gp.mutation(Individual(x))
        assert nx.genotype.depth > x.depth