from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Annotated

from geneticengine.core.decorators import abstract

from geneticengine.analysis.production_analysis import count_productions
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.tree.initializations import grow_method, pi_grow_method
from geneticengine.core.representations.tree.treebased import random_node
from geneticengine.core.representations.grammatical_evolution.ge import phenotype_to_genotype
from geneticengine.core.representations.grammatical_evolution.ge import create_tree
from geneticengine.core.utils import get_arguments
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
class LeafLiteral(Root):
    name: int


@dataclass
class Rec(Root):
    n1: Root
    n2: Root


@dataclass
class RecAlt(Root):
    l: Leaf


def contains_type(t, ty: type):
    if isinstance(t, ty):
        return True
    elif isinstance(t, list):
        for el in t:
            if contains_type(el, ty):
                return True
    else:
        for argn, argt in get_arguments(t):
            if contains_type(getattr(t, argn), ty):
                return True
    return False


class TestPhenotypeToGenotype:
    def test_rec(self):
        r = RandomSource(seed=1)
        g = extract_grammar([Leaf, Rec, RecAlt], RootSuper)
        max_depth = 15
        x = random_node(r, g, max_depth, RootSuper, method=pi_grow_method)
        ind1 = phenotype_to_genotype(r=r, g=g, p=x, depth=max_depth)
        x1 = create_tree(g, ind1, max_depth, initialization_mode=grow_method)
        ind2 = phenotype_to_genotype(r=r, g=g, p=x, depth=max_depth + 5)
        x2 = create_tree(g, ind2, max_depth + 5, initialization_mode=grow_method)
        phenotype_to_genotype(r=r, g=g, p=x, depth=max_depth + 5)
        x3 = create_tree(g, ind2, max_depth + 5, initialization_mode=grow_method)
        assert x == x1
        assert x == x2
        assert x1 == x2
        assert x == x3

    def test_var_range(self):
        r = RandomSource(seed=1)
        g = extract_grammar([Leaf, Rec, RecAlt, LeafVar], RootSuper)
        max_depth = 10
        x = random_node(r, g, max_depth, RootSuper, method=pi_grow_method)
        ind1 = phenotype_to_genotype(r=r, g=g, p=x, depth=max_depth)
        x1 = create_tree(g, ind1, max_depth, initialization_mode=grow_method)
        ind2 = phenotype_to_genotype(r=r, g=g, p=x, depth=max_depth + 5)
        x2 = create_tree(g, ind2, max_depth + 5, initialization_mode=grow_method)
        phenotype_to_genotype(r=r, g=g, p=x, depth=max_depth + 5)
        x3 = create_tree(g, ind2, max_depth + 5, initialization_mode=grow_method)
        assert contains_type(x, LeafVar)
        assert x == x1
        assert x == x2
        assert x1 == x2
        assert x == x3

    def test_literal_range(self):
        r = RandomSource(seed=1)
        g = extract_grammar([Leaf, Rec, RecAlt, LeafVar, LeafLiteral], RootSuper)
        max_depth = 10
        x = random_node(r, g, max_depth, RootSuper, method=pi_grow_method)
        ind1 = phenotype_to_genotype(r=r, g=g, p=x, depth=max_depth)
        x1 = create_tree(g, ind1, max_depth, initialization_mode=grow_method)
        ind2 = phenotype_to_genotype(r=r, g=g, p=x, depth=max_depth + 5)
        x2 = create_tree(g, ind2, max_depth + 5, initialization_mode=grow_method)
        phenotype_to_genotype(r=r, g=g, p=x, depth=max_depth + 5)
        x3 = create_tree(g, ind2, max_depth + 5, initialization_mode=grow_method)
        assert contains_type(x, LeafLiteral)
        assert count_productions(x, g) == count_productions(x1, g)
        assert count_productions(x, g) == count_productions(x2, g)
        assert count_productions(x1, g) == count_productions(x2, g)
        assert count_productions(x, g) == count_productions(x3, g)
