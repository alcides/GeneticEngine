from __future__ import annotations
from typing import Iterator

from geneticengine.solutions.individual import Individual
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.problems import Problem
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import Representation
from geneticengine.evaluation import Evaluator


class NoveltyStep(GeneticStep):
    """Creates new individuals for the population."""

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
        for _ in range(target_size):
            yield Individual(
                representation.create_genotype(random),
                representation,
            )
