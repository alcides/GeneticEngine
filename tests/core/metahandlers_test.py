from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Annotated, Union

import numpy as np

from geneticengine.grammar.grammar import extract_grammar
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.representations.tree.treebased import random_node
from geneticengine.grammar.metahandlers.floats import FloatRange
from geneticengine.grammar.metahandlers.ints import IntervalRange
from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.grammar.metahandlers.lists import ListSizeBetween
from geneticengine.grammar.metahandlers.strings import StringSizeBetween, WeightedStringHandler
from geneticengine.grammar.metahandlers.vars import VarRange


class Root(ABC):
    pass


@dataclass
class IntRangeM(Root):
    x: Annotated[int, IntRange[9, 10]]


@dataclass
class UnionIntRangeM(Root):
    x: Union[Annotated[int, IntRange[0, 10]], Annotated[int, IntRange[20, 30]]]


@dataclass
class IntervalRangeM(Root):
    x: Annotated[
        tuple[int, int],
        IntervalRange(
            minimum_length=5,
            maximum_length=10,
            maximum_top_limit=100,
        ),
    ]


@dataclass
class FloatRangeM(Root):
    x: Annotated[float, FloatRange[9.0, 10.0]]


@dataclass
class VarRangeM(Root):
    x: Annotated[str, VarRange(["x", "y", "z"])]


@dataclass
class ListRangeM(Root):
    x: Annotated[list[int], ListSizeBetween[3, 4]]


string_options = ["a", "t", "c", "g"]
string_meta = StringSizeBetween(3, 7, string_options)


@dataclass
class StringM(Root):
    x: Annotated[str, string_meta]


@dataclass
class WeightedStringHandlerM(Root):
    x: Annotated[
        str,
        WeightedStringHandler(
            np.array(
                [
                    [-0.01, 0.0425531914894, 0.01, 0.937446808511],
                    [0.97, 0.01, 0.01, 0.01],
                    [0.0212765957447, 0.01, 0.958723404255, 0.01],
                    [
                        0.106382978723,
                        0.0212765957447,
                        0.787234042553,
                        0.0851063829787,
                    ],
                    [0.533191489362, 0.01, 0.01, 0.446808510638],
                ],
            ),
            ["A", "C", "G", "T"],
        ),
    ]

@dataclass
class Number:
    number: int


@dataclass
class GenerateNumber(Root):
    x: Annotated[Number, VarRange([Number(0), Number(1), Number(2)])]

class TestMetaHandler:
    def test_int(self):
        r = NativeRandomSource(seed=1)
        g = extract_grammar([IntRangeM], Root)
        decider = MaxDepthDecider(r, g, 3)
        n = random_node(r, g, Root, decider=decider)
        assert isinstance(n, IntRangeM)
        assert 9 <= n.x <= 10
        assert isinstance(n, Root)

    def test_float(self):
        r = NativeRandomSource(seed=1)
        g = extract_grammar([FloatRangeM], Root)
        decider = MaxDepthDecider(r, g, 3)
        n = random_node(r, g, Root, decider=decider)
        assert isinstance(n, FloatRangeM)
        assert 9.0 <= n.x <= 10.0
        assert isinstance(n, Root)

    def test_var(self):
        r = NativeRandomSource(seed=1)
        g = extract_grammar([VarRangeM], Root)
        decider = MaxDepthDecider(r, g, 3)
        n = random_node(r, g, Root, decider=decider)
        assert isinstance(n, VarRangeM)
        assert n.x in ["x", "y", "z"]
        assert isinstance(n, Root)

    def test_list(self):
        r = NativeRandomSource(seed=1)
        g = extract_grammar([ListRangeM], Root)
        decider = MaxDepthDecider(r, g, 4)
        n = random_node(r, g, Root, decider=decider)
        assert isinstance(n, ListRangeM)
        assert 3 <= len(n.x) <= 4
        assert isinstance(n, Root)

    def test_string(self):
        r = NativeRandomSource(seed=1)
        g = extract_grammar([StringM], Root)
        decider = MaxDepthDecider(r, g, 3)
        n = random_node(r, g, Root, decider=decider)
        assert isinstance(n.x, str)
        assert len(n.x) >= 3
        assert len(n.x) <= 7
        assert all(x in string_options for x in n.x)
        for _ in range(10):
            n.x = string_meta.mutate(r, g, random_node, 2, str, n.x)
            assert isinstance(n.x, str)
            assert len(n.x) >= 3
            assert len(n.x) <= 7
            assert all(x in string_options for x in n.x)

        for _ in range(10):
            n.x = string_meta.crossover(r, g, [StringM("ccc"), StringM("cccc")], "x", str, n.x)
            assert isinstance(n.x, str)
            assert len(n.x) >= 3
            assert len(n.x) <= 7
            assert all(x in string_options for x in n.x)

    def test_weightedstrings(self):
        r = NativeRandomSource(seed=1)
        g = extract_grammar([WeightedStringHandlerM], Root)
        decider = MaxDepthDecider(r, g, 3)
        n = random_node(r, g, Root, decider=decider)
        assert isinstance(n, WeightedStringHandlerM)
        assert all(_x in ["A", "C", "G", "T"] for _x in n.x)
        assert isinstance(n, Root)

    def test_intervalrange(self):
        r = NativeRandomSource(seed=1)
        g = extract_grammar([IntervalRangeM], Root)
        decider = MaxDepthDecider(r, g, 4)
        n = random_node(r, g, Root, decider=decider)

        assert isinstance(n, IntervalRangeM)
        assert 5 < n.x[1] - n.x[0] < 10 and n.x[1] < 100
        assert isinstance(n, Root)

    def test_union_int(self):
        r = NativeRandomSource(seed=1)
        g = extract_grammar([UnionIntRangeM], Root)
        decider = MaxDepthDecider(r, g, 3)
        for _ in range(100):
            n = random_node(r, g, Root, decider=decider)
            assert isinstance(n, UnionIntRangeM)
            assert (0 <= n.x <= 10) or (20 <= n.x <= 30)
            assert isinstance(n, Root)

    def test_var_with_custom_class(self):
        r = NativeRandomSource(seed=1)
        g = extract_grammar([GenerateNumber], Root)
        decider = MaxDepthDecider(r, g, 3)
        n = random_node(r, g, Root, decider=decider)
        assert isinstance(n, GenerateNumber)
        assert n.x.number in [1, 2, 3]
        assert isinstance(n, Root)
