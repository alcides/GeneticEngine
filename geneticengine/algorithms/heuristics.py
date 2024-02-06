from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from geneticengine.algorithms.api import SynthesisAlgorithm
from geneticengine.evaluation.api import Evaluator

from geneticengine.evaluation.budget import SearchBudget
from geneticengine.evaluation.recorder import SingleObjectiveProgressTracker
from geneticengine.grammar.grammar import Grammar
from geneticengine.problems import Problem
from geneticengine.problems.helpers import best_individual
from geneticengine.random.sources import NativeRandomSource
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import Representation, SolutionRepresentation

from geneticengine.solutions.individual import Individual


# TODO: Remove
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
        self.evaluator.evaluate(problem, individuals)
        bi = best_individual(individuals, problem)
        return bi


class HeuristicSearch(SynthesisAlgorithm):
    """Randomly generates new solutions and keeps the best one."""

    random: RandomSource

    def __init__(
        self,
        problem: Problem,
        budget: SearchBudget,
        representation: SolutionRepresentation,
        random: RandomSource = None,
        recorder: SingleObjectiveProgressTracker | None = None,
    ):
        super().__init__(problem, budget, representation, recorder)
        if random is None:
            self.random = NativeRandomSource(0)
        else:
            self.random = random
