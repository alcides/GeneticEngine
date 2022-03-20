from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Annotated
from typing import List
from typing import Type
from unittest import skip

from scipy import rand

from geneticengine.algorithms.gp.gp import GP
from geneticengine.algorithms.gp.individual import Individual
from geneticengine.core.decorators import abstract
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.tree.treebased import Grow
from geneticengine.core.representations.tree.treebased import PI_Grow
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


@dataclass
class RecAlt(Root):
    l: Leaf


def contains_type(t, ty: type, globalns):
    if isinstance(t, ty):
        return True
    elif isinstance(t, list):
        for el in t:
            if contains_type(el, ty, globalns):
                return True
    else:
        for (argn, argt) in get_arguments(t, globalns):
            if contains_type(getattr(t, argn), ty, globalns):
                return True
    return False


class TestGrammar:
    def test_root(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar(Root, globals(), [Leaf])
        x = random_node(r, g, 2, Root)
        assert isinstance(x, Leaf)
        assert isinstance(x, Root)

    def test_rec(self):
        r = RandomSource(seed=1)
        g: Grammar = extract_grammar(Root, globals(), [Leaf, Rec])
        x = random_node(r, g, 10, Root, method=PI_Grow)
        # print(x) -- Leaf()
        assert isinstance(x, Rec)
        assert isinstance(x, Root)

    def test_rec_alt(self):
        r = RandomSource(seed=245)
        g: Grammar = extract_grammar(Root, globals(), [Leaf, Rec, RecAlt])
        x = random_node(
            r,
            g,
            max_depth=15,
            starting_symbol=Root,
            method=PI_Grow,
        )
        assert contains_type(x, RecAlt, globals())
        assert isinstance(x, Root)

    @skip("Reevaluate what this test does")
    def test_depth_increases(self):
        g: Grammar = extract_grammar(Root, globals(), [Leaf, Rec])

        x = random_node(
            RandomSource(3),
            g,
            max_depth=2,
            starting_symbol=Root,
            method=PI_Grow,
        )
        assert isinstance(x, Leaf)
        assert isinstance(x, Root)

        gp = GP(
            g,
            evaluation_function=lambda x: x.depth,
            randomSource=RandomSource,
            max_depth=5,
            seed=5,
        )

        nx = gp.mutation(Individual(x))
        assert nx.genotype.depth > x.depth
