from __future__ import annotations

from typing import Any, Iterator

from geneticengine.solutions.individual import Individual
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.problems import Problem
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import RepresentationWithMutation, Representation
from geneticengine.evaluation import Evaluator


class GenericMutationStep(GeneticStep):
    """Applies a mutation to individuals with a given probability."""

    def __init__(
        self,
        probability: float = 1,
    ):
        self.probability = probability

    def iterate(
        self,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random: RandomSource,
        population: Iterator[Individual],
        target_size: int,
        generation: int,
    ) -> Iterator[Individual]:
        assert isinstance(representation, RepresentationWithMutation)
        for index, ind in enumerate(population):
            if index < target_size:
                v = random.random_float(0, 1)
                if v <= self.probability:
                    mutated = representation.mutate(random, ind.genotype)
                    nind = self.wrap(representation, mutated)
                    yield nind
                else:
                    yield ind

    def wrap(self, representation: Representation, genotype: Any) -> Individual:
        return Individual(
            genotype=genotype,
            representation=representation,
        )
