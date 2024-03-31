from __future__ import annotations
import logging
from typing import Iterator

from geneticengine.solutions.individual import Individual
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.problems import Problem
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import RepresentationWithCrossover, Representation
from geneticengine.evaluation import Evaluator

logger = logging.getLogger(__name__)


class GenericCrossoverStep(GeneticStep):
    """Changes the population by crossing individuals two-by-two together,
    according to a given probability."""

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
        assert isinstance(representation, RepresentationWithCrossover)
        npopulation = list(population)
        for i in range(target_size // 2):
            j = i % len(npopulation)
            ind1, ind2 = npopulation[j], npopulation[j + 1]  # todo: select individuals using a selection method
            assert isinstance(ind1, Individual)
            assert isinstance(ind2, Individual)
            v = random.random_float(0, 1)
            if v <= self.probability:
                (n1, n2) = self.crossover(random, ind1, ind2, representation=representation)
            else:
                (n1, n2) = (ind1, ind2)
            yield n1
            yield n2

        if (target_size // 2) * 2 < target_size:
            yield npopulation[0]

    def crossover(
        self,
        random: RandomSource,
        individual1: Individual,
        individual2: Individual,
        representation: Representation,
    ):
        assert isinstance(representation, RepresentationWithCrossover)
        logger.info(f"Crossing-over {individual1.genotype} with {individual2.genotype}")
        (g1, g2) = representation.crossover(
            random,
            individual1.genotype,
            individual2.genotype,
        )
        return (
            Individual(g1, representation=representation),
            Individual(g2, representation=representation),
        )
