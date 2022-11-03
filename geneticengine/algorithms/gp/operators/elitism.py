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
        population_copy = population.copy()
        population_copy.sort(
            key=lambda ind: ind.evaluate(problem, representation.genotype_to_phenotype),
        )
        return population_copy[:target_size]
