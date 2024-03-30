from typing import Iterator
from geneticengine.solutions.individual import Individual
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.problems import Problem
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import Representation
from geneticengine.evaluation import Evaluator


class EvaluateStep(GeneticStep):
    """Evaluates the complete population."""

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
        evaluator.evaluate(problem, population)
        yield from population
