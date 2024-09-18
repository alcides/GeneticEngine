from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from geneticengine.problems import SingleObjectiveProblem


from geneticengine.grammar.decorators import abstract
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import (
    FullDecider,
    PositionIndependentGrowDecider,
    ProgressivelyTerminalDecider,
)
from geneticengine.representations.tree.operators import FullInitializer, GrowInitializer
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.grammar.metahandlers.floats import FloatRange
from geneticengine.grammar.metahandlers.ints import IntervalRange
from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.grammar.metahandlers.lists import ListSizeBetween
from geneticengine.grammar.metahandlers.vars import VarRange


@abstract
class A:
    pass


@dataclass(unsafe_hash=True)
class B(A):
    i: int
    s: str
    m: list[int]
    ia: Annotated[int, IntRange[9, 10]]
    tau: Annotated[
        tuple[int, int],
        IntervalRange(
            minimum_length=5,
            maximum_length=10,
            maximum_top_limit=100,
        ),
    ]
    fa: Annotated[float, FloatRange[9.0, 10.0]]
    sa: Annotated[str, VarRange(["x", "y", "z"])]
    la: Annotated[list[int], ListSizeBetween[3, 7]]


@dataclass
class C(A):
    one: A
    two: A


class TestInitializers:
    def test_full(self):
        target_size = 10
        target_depth = 10

        g = extract_grammar([B, C], A)
        f = FullInitializer(max_depth=target_depth)
        p = SingleObjectiveProblem(lambda x: 3)
        rs = NativeRandomSource(5)
        repr = TreeBasedRepresentation(grammar=g, decider=FullDecider(rs, g, target_depth))

        population = list(f.initialize(p, repr, rs, target_size))
        assert len(population) == target_size
        for ind in population:
            assert ind.get_phenotype().gengy_distance_to_term == target_depth

    def test_pi_grow(self):
        target_size = 10
        target_depth = 10

        g = extract_grammar([B, C], A)
        f = GrowInitializer()
        p = SingleObjectiveProblem(lambda x: 3)
        rs = NativeRandomSource(5)
        repr = TreeBasedRepresentation(grammar=g, decider=PositionIndependentGrowDecider(rs, g, target_depth))

        population = list(f.initialize(p, repr, rs, target_size))
        assert len(population) == target_size
        for ind in population:
            assert ind.get_phenotype().gengy_distance_to_term <= target_depth

    def test_progressive(self):
        target_size = 10

        g = extract_grammar([B, C], A)
        f = GrowInitializer()
        p = SingleObjectiveProblem(lambda x: 3)
        rs = NativeRandomSource(5)
        repr = TreeBasedRepresentation(grammar=g, decider=ProgressivelyTerminalDecider(rs, g))

        population = list(f.initialize(p, repr, rs, target_size))
        assert len(population) == target_size
        for ind in population:
            assert ind.get_phenotype()
