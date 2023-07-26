from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Annotated

import pytest
from geneticengine.core.decorators import abstract

from geneticengine.core.grammar import extract_grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.tree.initializations import grow_method, pi_grow_method
from geneticengine.core.representations.tree.treebased import random_node
from geneticengine.core.representations.grammatical_evolution.ge import phenotype_to_genotype
from geneticengine.core.representations.grammatical_evolution.ge import create_tree
from geneticengine.core.representations.grammatical_evolution.ge import Genotype
from geneticengine.metahandlers.vars import VarRange


class RootSuper(ABC):
    pass

@abstract
class Root(RootSuper):
    pass

@dataclass
class Leaf(Root):
    pass

@dataclass
class LeafVar(Root):
    name: Annotated[str, VarRange(["x", "y", "z"])]


@dataclass
class Rec(Root):
    n1: Root
    n2: Root


@dataclass
class RecAlt(Root):
    l: Leaf


class Useless(Root):
    def __init__(self, a):
        pass  # a does not have a type


class TestGrammar:

    def test_rec(self):
        r = RandomSource(seed=1)
        g = extract_grammar([Leaf, Rec, RecAlt], RootSuper)
        max_depth = 15
        x = random_node(r, g, max_depth, RootSuper, method=pi_grow_method)
        ind1 = phenotype_to_genotype(g=g, p=x, depth=max_depth)
        x1 = create_tree(g, ind1, max_depth, initialization_mode=grow_method)
        ind2 = phenotype_to_genotype(g=g, p=x, depth=max_depth + 5)
        x2 = create_tree(g, ind2, max_depth + 5, initialization_mode=grow_method)
        assert x == x1
        assert x == x2
        assert x1 == x2

    def test_var_range(self):
        r = RandomSource(seed=1)
        g = extract_grammar([Leaf, Rec, RecAlt, LeafVar], RootSuper)
        max_depth = 10
        x = random_node(r, g, max_depth, RootSuper, method=pi_grow_method)
        ind1 = phenotype_to_genotype(g=g, p=x, depth=max_depth)
        x1 = create_tree(g, ind1, max_depth, initialization_mode=grow_method)
        ind2 = phenotype_to_genotype(g=g, p=x, depth=max_depth + 5)
        x2 = create_tree(g, ind2, max_depth + 5, initialization_mode=grow_method)
        assert x == x1
        assert x == x2
        assert x1 == x2


test = TestGrammar
test.test_rec(test)
test.test_var_range(test)
