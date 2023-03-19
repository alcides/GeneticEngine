from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated
from geneticengine.core.problems import SingleObjectiveProblem


from geneticengine.core.decorators import abstract
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.tree.operators import FullInitializer, GrowInitializer
from geneticengine.core.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.metahandlers.floats import FloatRange
from geneticengine.metahandlers.ints import IntervalRange
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.lists import ListSizeBetween
from geneticengine.metahandlers.vars import VarRange


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
        f = FullInitializer()
        p = SingleObjectiveProblem(lambda x: 3)
        repr = TreeBasedRepresentation(grammar=g, max_depth=target_depth)
        rs = RandomSource(5)

        population = f.initialize(p, repr, rs, target_size)
        assert len(population) == target_size
        for ind in population:
            assert ind.get_phenotype().gengy_distance_to_term == target_depth

    def test_pi_grow(self):
        target_size = 10
        target_depth = 10

        g = extract_grammar([B, C], A)
        f = GrowInitializer()
        p = SingleObjectiveProblem(lambda x: 3)
        repr = TreeBasedRepresentation(grammar=g, max_depth=target_depth)
        rs = RandomSource(5)

        population = f.initialize(p, repr, rs, target_size)
        assert len(population) == target_size
        for ind in population:
            assert ind.get_phenotype().gengy_distance_to_term <= target_depth
