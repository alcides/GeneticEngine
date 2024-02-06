from __future__ import annotations

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
        random_source: RandomSource,
        population: list[Individual],
        target_size: int,
        generation: int,
    ) -> list[Individual]:
        return [
            Individual(
                representation.create_individual(random_source, depth=None),
                representation.genotype_to_phenotype,
            )
            for _ in range(target_size)
        ]
