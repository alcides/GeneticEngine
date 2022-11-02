from __future__ import annotations

from geneticengine.algorithms.gp.individual import Individual
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.core.problems import Problem
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import Representation


class ElitismStep(GeneticStep):
    """Selects the best individuals from the population."""

    def iterate(
        self,
        problem: Problem,
        representation: Representation,
        random_source: Source,
        population: list[Individual],
        target_size: int,
    ) -> list[Individual]:

        fitnesses = [problem.evaluate(x) for x in population]

        if isinstance(problem, SingleObjectiveProblem):
            assert all(isinstance(x, float) for x in fitnesses)
        else:
            assert all(isinstance(x, list) for x in fitnesses)

        population_copy = population.copy()
        population_copy.sort(key=lambda x: problem.overall_fitness(x))
        return population_copy[:target_size]
