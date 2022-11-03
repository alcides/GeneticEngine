from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from operator import attrgetter
from typing import Any
from typing import Callable

from geneticengine.algorithms.gp.individual import Individual
from geneticengine.core.grammar import Grammar
from geneticengine.core.problems import FitnessType
from geneticengine.core.problems import MultiObjectiveProblem
from geneticengine.core.problems import Problem
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.api import Representation


class Heuristics(ABC):
    """An Abstract class that gp.py, hill_climbing.py and random_search extends
    to.

    Args:
        grammar (Grammar): The grammar used to guide the search.
        representation (Representation): The individual representation used by the GP program. The default is TreeBasedRepresentation.
        problem (Problem): The problem we are solving. Either a SingleObjectiveProblem or a MultiObjectiveProblem.
        randomSource (Callable[[int], RandomSource]): The random source function used by the program. Should take in an integer, representing the seed, and return a RandomSource.
    """

    grammar: Grammar
    representation: Representation
    problem: Problem
    random: RandomSource

    def __init__(
        self,
        grammar: Grammar,
        representation: Representation,
        problem: Problem,
        randomSource: Callable[[int], RandomSource],
        seed: int,
    ):
        self.problem: Problem = problem
        self.grammar = grammar
        self.representation = representation
        self.random = randomSource(seed)

    @abstractmethod
    def evolve(self):
        ...

    def get_best_individual(
        self,
        p: Problem,
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
        best_individual: Individual
        fitnesses = [x.evaluate(self.problem) for x in individuals]

        if isinstance(p, SingleObjectiveProblem):
            assert all(isinstance(x, float) for x in fitnesses)
            if p.minimize:
                best_individual = min(individuals, key=attrgetter("fitness"))
            else:
                best_individual = max(individuals, key=attrgetter("fitness"))

        elif isinstance(p, MultiObjectiveProblem):
            assert all(isinstance(x, list) for x in fitnesses)

            def single_criteria(i: Individual) -> float:
                assert isinstance(p.minimize, list)
                assert isinstance(i.fitness, list)
                return sum((m and -f or f) for (f, m) in zip(i.fitness, p.minimize))

            if p.best_individual_criteria_function is None:
                best_individual = max(individuals, key=single_criteria)
            else:
                fun = p.best_individual_criteria_function
                best_individual = max(
                    individuals,
                    key=lambda ind: fun(ind.get_phenotype()),
                )

        return best_individual

    # this only works with SingleObjectiveProblem
    def keyfitness(self):
        if self.problem.minimize:
            return lambda x: x.evaluate(self.problem)
        else:
            return lambda x: -x.evaluate(self.problem)
