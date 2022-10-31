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
    """
    An Abstract class that gp.py, hill_climbing.py and random_search extends to

    Args:
        grammar (Grammar): The grammar used to guide the search.
        representation (Representation): The individual representation used by the GP program. The default is treebased_representation.
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
    def evolve(self, verbose):
        ...

    def get_best_individual(
        self,
        p: Problem,
        individuals: list[Individual],
    ) -> Individual:
        """
        The get_best_individual is a method that that returns the best individual of a population

        Args:
            p (Problem): the problem we are trying to solve
            individuals (list[Individual]): the list of individuals where we're going to search for the best one

        Returns:
            returns an Individual
        """
        best_individual: Individual
        fitnesses = [self.evaluate(x) for x in individuals]

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
                best_individual = max(
                    individuals,
                    key=p.best_individual_criteria_function,
                )

        return best_individual

    # this only works with SingleObjectiveProblem
    def keyfitness(self):
        if self.problem.minimize:
            return lambda x: self.evaluate(x)
        else:
            return lambda x: -self.evaluate(x)

    def evaluate(self, individual: Individual) -> FitnessType:
        """
        The evaluate is a methoad that is used to evaluate individual fitness

        Args:
            individual (Individual): Individual that we are evaluating the fitness

        Returns:
            FitnessType: The FitnessType of the individual, either a float or a list[float]

        """
        if individual.fitness is None:
            phenotype = self.representation.genotype_to_phenotype(
                self.grammar,
                individual.genotype,
            )
            individual.fitness = self.problem.evaluate(phenotype)
        return individual.fitness

    def create_individual(self, depth: int) -> Individual:
        """
        The create_individual is a methoad that is used to create a new individual

        Args:
            depth: number of

        Returns:
            Individual

        """
        genotype = self.representation.create_individual(
            r=self.random,
            g=self.grammar,
            depth=depth,
        )
        return Individual(
            genotype=genotype,
            fitness=None,
        )
