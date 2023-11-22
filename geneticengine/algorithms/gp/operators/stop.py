from __future__ import annotations

from geneticengine.algorithms.gp.individual import Individual
from geneticengine.algorithms.gp.structure import StoppingCriterium
from geneticengine.core.evaluators import Evaluator
from geneticengine.core.fitness_helpers import best_individual
from geneticengine.core.problems import Fitness, Problem


class GenerationStoppingCriterium(StoppingCriterium):
    """Runs the evolution during a number of generations."""

    def __init__(self, max_generations: int):
        """Creates a limit for the evolution, based on the number of
        generations.

        Arguments:
            max_generations (int): Number of generations to execute
        """
        self.max_generations = max_generations

    def is_ended(
        self,
        problem: Problem,
        population: list[Individual],
        generation: int,
        elapsed_time: float,
        evaluator: Evaluator,
    ) -> bool:
        return generation >= self.max_generations


class TimeStoppingCriterium(StoppingCriterium):
    """Runs the evolution during a given amount of time.

    Note that termination is not pre-emptive. If the fitness function is
    slow, this might take more than the pre-specified time.
    """

    def __init__(self, max_time: int):
        """Creates a limit for the evolution, based on the execution time.

        Arguments:
            max_time (int): Maximum time in seconds to run the evolution
        """
        self.max_time = max_time

    def is_ended(
        self,
        problem: Problem,
        population: list[Individual],
        generation: int,
        elapsed_time: float,
        evaluator: Evaluator,
    ) -> bool:
        return elapsed_time >= self.max_time


class EvaluationLimitCriterium(StoppingCriterium):
    """Runs the evolution with a fixed budget for evaluations."""

    def __init__(self, max_evaluations: int):
        """Creates a limit for the evolution, based on the budget for
        evaluation.

        Arguments:
            max_evaluations (int): Maximum number of evaluations
        """
        self.max_evaluations = max_evaluations

    def is_ended(
        self,
        problem: Problem,
        population: list[Individual],
        generation: int,
        elapsed_time: float,
        evaluator: Evaluator,
    ) -> bool:
        return evaluator.get_count() >= self.max_evaluations


class SingleFitnessTargetStoppingCriterium(StoppingCriterium):
    """Stops the evolution when the fitness gets to a given value."""

    def __init__(self, target_fitness: float, epsilon=0):
        self.target_fitness = target_fitness
        self.epsilon = epsilon

    def is_ended(
        self,
        problem: Problem,
        population: list[Individual],
        generation: int,
        elapsed_time: float,
        evaluator: Evaluator,
    ) -> bool:
        evaluator.eval(problem, population)
        best_fitness = best_individual(population, problem).get_fitness(problem)
        return (
            problem.is_better(best_fitness, Fitness(self.target_fitness, []))
            or abs(best_fitness.maximizing_aggregate - self.target_fitness) <= self.epsilon
        )


class AllFitnessTargetStoppingCriterium(StoppingCriterium):
    """Stops the evolution when the fitness gets to a given value."""

    def __init__(self, target_fitnesses: list[float]):
        self.target_fitnesses = target_fitnesses

    def is_ended(
        self,
        problem: Problem,
        population: list[Individual],
        generation: int,
        elapsed_time: float,
        evaluator: Evaluator,
    ) -> bool:
        evaluator.eval(problem, population)
        best_fitness = best_individual(population, problem).get_fitness(problem)

        def compare_fitness(a, b, minimize):
            return a < b if minimize else a > b

        if isinstance(problem.minimize, list):
            return all(
                compare_fitness(a, b, m)
                for a, b, m in zip(best_fitness.fitness_components, self.target_fitnesses, problem.minimize)
            )
        elif isinstance(problem.minimize, bool):
            return all(
                compare_fitness(a, b, problem.minimize)
                for a, b in zip(best_fitness.fitness_components, self.target_fitnesses)
            )

        assert False


class AnyOfStoppingCriterium(StoppingCriterium):
    """Stops the evolution when any of the two stopping criteria is done."""

    def __init__(self, one: StoppingCriterium, other: StoppingCriterium):
        self.one = one
        self.other = other

    def is_ended(
        self,
        problem: Problem,
        population: list[Individual],
        generation: int,
        elapsed_time: float,
        evaluator: Evaluator,
    ) -> bool:
        return self.one.is_ended(
            problem=problem,
            population=population,
            generation=generation,
            elapsed_time=elapsed_time,
            evaluator=evaluator,
        ) or self.other.is_ended(
            problem=problem,
            population=population,
            generation=generation,
            elapsed_time=elapsed_time,
            evaluator=evaluator,
        )
