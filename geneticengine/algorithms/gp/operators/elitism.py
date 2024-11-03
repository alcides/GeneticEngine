from __future__ import annotations
from typing import Iterator

from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.problems.helpers import sort_population
from geneticengine.problems import Problem
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import Representation
from geneticengine.evaluation import Evaluator
from geneticengine.solutions.individual import PhenotypicIndividual


class ElitismStep(GeneticStep):
    """Selects the best individuals from the population."""

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
        evaluator.evaluate(problem, population)
        # TODO: We do not need to sort here.
        new_population = sort_population(list(population), problem)
        yield from new_population[:target_size]
