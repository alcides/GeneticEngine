from abc import ABC, abstractmethod
from typing import Optional
from geneticengine.evaluation.budget import SearchBudget

from geneticengine.evaluation.recorder import SingleObjectiveProgressTracker
from geneticengine.evaluation.sequential import SequentialEvaluator
from geneticengine.problems import Problem
from geneticengine.representations.api import Representation
from geneticengine.solutions.individual import Individual


class SynthesisAlgorithm(ABC):
    tracker: SingleObjectiveProgressTracker
    problem: Problem
    budget: SearchBudget
    representation: Representation

    def __init__(
        self,
        problem: Problem,
        budget: SearchBudget,
        representation: Representation,
        tracker: Optional[SingleObjectiveProgressTracker] = None,
    ):
        self.problem = problem
        self.budget = budget
        self.representation = representation

        if tracker is None:
            self.tracker = SingleObjectiveProgressTracker(problem, SequentialEvaluator())
        else:
            self.tracker = tracker

    def is_done(self) -> bool:
        """Whether the synthesis should stop, or not."""
        return self.budget.is_done(self.tracker)

    def get_best_solution(self) -> Individual:
        """Returns the best solution found during the search."""
        return self.tracker.get_best_individual()

    @abstractmethod
    def search(self):
        ...
