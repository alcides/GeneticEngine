from __future__ import annotations

from geneticengine.solutions.individual import Individual
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.problems.helpers import sort_population
from geneticengine.problems import Problem
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import Representation
from geneticengine.evaluation import Evaluator


class ElitismStep(GeneticStep):
    """Selects the best individuals from the population."""

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
        evaluator.evaluate(problem, population)
        new_population = sort_population(population, problem)
        return new_population[:target_size]
