from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Annotated, Iterator

import pytest

from geneticengine.algorithms.gp.gp import default_generic_programming_step
from geneticengine.representations.api import Representation
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.solutions.individual import PhenotypicIndividual
from geneticengine.algorithms.gp.operators.combinators import SequenceStep
from geneticengine.algorithms.gp.operators.crossover import GenericCrossoverStep
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.algorithms.gp.operators.selection import TournamentSelection
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.evaluation import Evaluator
from geneticengine.evaluation.sequential import SequentialEvaluator
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.problems import Fitness, Problem, SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource, RandomSource
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.grammar.metahandlers.lists import ListSizeBetween


class Alt(ABC):
    pass


@dataclass
class Sub(Alt):
    a: int
    b: int


@dataclass
class Base:
    li: Annotated[list[Alt], ListSizeBetween(1, 2)]


class CacheFitness(GeneticStep):
    def iterate(
        self,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random: RandomSource,
        population: Iterator[PhenotypicIndividual],
        target_size: int,
        generation: int,
    ) -> Iterator[PhenotypicIndividual]:
        for ind in population:
            ind.set_fitness(problem, Fitness([]))
        return population


class TestPreCache:
    @pytest.mark.parametrize(
        "test_step",
        [
            GenericMutationStep(1.0),
            GenericCrossoverStep(1.0),
            TournamentSelection(3),
            default_generic_programming_step(),
        ],
    )
    def test_immutability(self, test_step):
        g = extract_grammar([Sub], Base)
        rep = TreeBasedRepresentation(g, decider=MaxDepthDecider(NativeRandomSource(), g, max_depth=10))
        r = NativeRandomSource(3)

        def fitness_function(x):
            assert False

        problem = SingleObjectiveProblem(fitness_function=fitness_function)
        population_size = 1000

        initial_population : list[PhenotypicIndividual] = [
            PhenotypicIndividual(genotype=rep.create_genotype(r), representation=rep) for _ in range(population_size)
        ]

        def encode_population(pop: list[PhenotypicIndividual]) -> list[str]:
            return [str(ind.genotype) for ind in pop]

        cpy = encode_population(initial_population)

        step = SequenceStep(CacheFitness(), default_generic_programming_step())

        for i in range(10):
            _ = step.apply(
                problem=problem,
                evaluator=SequentialEvaluator(),
                representation=rep,
                random=r,
                population=iter(initial_population),
                target_size=population_size,
                generation=i,
            )
        for a, b in zip(encode_population(initial_population), cpy):
            assert a == b
