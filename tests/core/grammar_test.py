from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

import pytest

from geneticengine.grammar.grammar import extract_grammar
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.representations.tree.treebased import random_node
from geneticengine.grammar.utils import get_arguments


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


class Useless(Root):
    def __init__(self, a):
        pass  # a does not have a type


@dataclass
class BasicList(Root):
    a: list[int]


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


class TestGrammar:
    def test_root(self):
        r = NativeRandomSource(seed=1)
        g = extract_grammar([Leaf], Root)
        decider = MaxDepthDecider(r, g, 2)
        x = random_node(r, g, Root, decider=decider)
        assert isinstance(x, Leaf)
        assert isinstance(x, Root)

    def test_rec(self):
        r = NativeRandomSource(seed=1)
        g = extract_grammar([Leaf, Rec], Root)
        decider = MaxDepthDecider(r, g, 10)
        x = random_node(r, g, Root, decider=decider)
        assert isinstance(x, Rec) or isinstance(x, Leaf)
        assert isinstance(x, Root)

    def test_rec_alt(self):
        r = NativeRandomSource(seed=245)
        g = extract_grammar([Leaf, Rec, RecAlt], Root)
        decider = MaxDepthDecider(r, g, 2)
        x = random_node(r, g, Root, decider=decider)
        assert isinstance(x, Root)

    def test_invalid_node(self):
        with pytest.raises(Exception):
            extract_grammar([Leaf, Rec, Useless], Root)

    def test_basic_list(self):
        extract_grammar([Leaf, Rec, BasicList], Root)
