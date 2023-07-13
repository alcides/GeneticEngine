from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from geneticengine.algorithms.gp.individual import Individual
from geneticengine.core.fitness_helpers import best_individual
from geneticengine.core.grammar import Grammar
from geneticengine.core.problems import Problem
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.api import Representation
from geneticengine.core.evaluators import Evaluator


class Heuristics(ABC):
    """An Abstract class that gp.py, hill_climbing.py and random_search extends
    to.

    Args:
        grammar (Grammar): The grammar used to guide the search.
        representation (Representation): The individual representation used by the GP program. The default is
            TreeBasedRepresentation.
        problem (Problem): The problem we are solving. Either a SingleObjectiveProblem or a MultiObjectiveProblem.
        randomSource (Callable[[int], RandomSource]): The random source function used by the program. Should take in an
            integer, representing the seed, and return a RandomSource.
    """

    grammar: Grammar
    representation: Representation
    problem: Problem
    random: RandomSource

    def __init__(
        self,
        representation: Representation,
        problem: Problem,
        evaluator: Evaluator,
    ):
        self.representation = representation
        self.problem: Problem = problem
        self.evaluator = evaluator

    @abstractmethod
    def evolve(self):
        ...

    def get_best_individual(
        self,
        problem: Problem,
        individuals: list[Individual],
    ) -> Individual:
        """The get_best_individual is a method that that returns the best
        individual of a population.

        Args:
            p (Problem): the problem we are trying to solve
            individuals (list[Individual]): the list of individuals where we're going to search for the best one

        Returns:
            returns an Individual
        """
        assert individuals
        self.evaluator.eval(problem, individuals)
        bi = best_individual(individuals, problem)
        return bi
