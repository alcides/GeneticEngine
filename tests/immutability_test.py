from __future__ import annotations
from abc import ABC

from dataclasses import dataclass
from typing import Annotated
from geneticengine.algorithms.gp.individual import Individual
from geneticengine.algorithms.gp.operators.crossover import GenericCrossoverStep
from geneticengine.algorithms.gp.operators.selection import TournamentSelection
from geneticengine.core.problems import SingleObjectiveProblem

import pytest

from geneticengine.core.decorators import abstract
from geneticengine.core.evaluators import SequentialEvaluator
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.metahandlers.floats import FloatRange
from geneticengine.metahandlers.ints import IntervalRange
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.lists import ListSizeBetween
from geneticengine.metahandlers.vars import VarRange
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.algorithms.gp.gp import default_generic_programming_step


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
class C:
    one: A
    two: A


class Alt(ABC):
    pass


@dataclass
class Three(Alt):
    a: int
    b: int


@dataclass
class One(Alt):
    li: Annotated[list[Three], ListSizeBetween(0, 5)]


@dataclass
class Two(Alt):
    li: list[Three]


class TestImmutability:
    def test_hash(self):
        g = extract_grammar([A, B], A)
        rep = TreeBasedRepresentation(g, max_depth=10)
        r = RandomSource(3)
        ind = rep.create_individual(r, 10)
        assert isinstance(hash(ind), int)

    @pytest.mark.parametrize(
        "test_step",
        [
            GenericMutationStep(1.0),
            GenericCrossoverStep(1.0),
            TournamentSelection(3),
            default_generic_programming_step(),
        ],
    )
    @pytest.mark.parametrize(
        "g",
        [
            extract_grammar([A, B, C], A),
            extract_grammar([One, Two, Three], Alt),
        ],
    )
    def test_immutability(self, test_step, g):
        rep = TreeBasedRepresentation(g, max_depth=10)
        r = RandomSource(3)
        problem = SingleObjectiveProblem(fitness_function=lambda x: 1)

        population_size = 1000

        initial_population = [
            Individual(genotype=rep.create_individual(r, 10), genotype_to_phenotype=rep.genotype_to_phenotype)
            for _ in range(population_size)
        ]

        def encode_population(pop: list[Individual]) -> list[str]:
            return [str(ind.genotype) for ind in pop]

        cpy = encode_population(initial_population)

        for i in range(10):
            _ = test_step.iterate(
                problem=problem,
                evaluator=SequentialEvaluator(),
                representation=rep,
                random_source=r,
                population=initial_population,
                target_size=population_size,
                generation=i,
            )
        for a, b in zip(encode_population(initial_population), cpy):
            assert a == b
