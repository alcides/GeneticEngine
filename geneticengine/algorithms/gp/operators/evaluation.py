from geneticengine.solutions.individual import Individual
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.problems.helpers import sort_population
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
        population: list[Individual],
        target_size: int,
        generation: int,
    ) -> list[Individual]:
        evaluator.evaluate(problem, population)
        # TODO: We do not need to sort here.
        new_population = sort_population(population, problem)
        return new_population
