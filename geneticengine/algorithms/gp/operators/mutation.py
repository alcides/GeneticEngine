from __future__ import annotations

from typing import Any

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
        population: list[Individual],
        target_size: int,
        generation: int,
    ) -> list[Individual]:
        assert isinstance(representation, RepresentationWithMutation)
        ret = []
        for index, ind in enumerate(population[:target_size]):
            assert isinstance(ind, Individual)
            v = random.random_float(0, 1)
            if v <= self.probability:
                mutated = representation.mutate(random, ind.genotype)
                nind = self.wrap(representation, mutated)
                ret.append(nind)
            else:
                ret.append(ind)

        return ret

    def wrap(self, representation: Representation, genotype: Any) -> Individual:
        return Individual(
            genotype=genotype,
            genotype_to_phenotype=representation.map,
        )
