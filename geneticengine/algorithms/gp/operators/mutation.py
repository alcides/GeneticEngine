from __future__ import annotations

import logging
from typing import Any, Iterator

from geneticengine.solutions.individual import PhenotypicIndividual
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.problems import Problem
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import RepresentationWithMutation, Representation
from geneticengine.evaluation import Evaluator

logger = logging.getLogger(__name__)


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
        population: Iterator[PhenotypicIndividual],
        target_size: int,
        generation: int,
    ) -> Iterator[PhenotypicIndividual]:
        assert isinstance(representation, RepresentationWithMutation)
        for index, ind in enumerate(population):
            if index < target_size:
                v = random.random_float(0, 1)
                if v <= self.probability:
                    logger.debug(f"Mutating {id(ind)}")
                    mutated = representation.mutate(random, ind.genotype)
                    nind = self.wrap(representation, mutated)
                    yield nind
                else:
                    yield ind

    def wrap(self, representation: Representation, genotype: Any) -> PhenotypicIndividual:
        return PhenotypicIndividual(
            genotype=genotype,
            representation=representation,
        )
