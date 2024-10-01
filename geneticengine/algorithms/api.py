from abc import ABC, abstractmethod
from typing import Optional
from geneticengine.evaluation.budget import SearchBudget

from geneticengine.evaluation.tracker import (
    MultiObjectiveProgressTracker,
    ProgressTracker,
    SingleObjectiveProgressTracker,
)
from geneticengine.evaluation.sequential import SequentialEvaluator
from geneticengine.problems import Problem, SingleObjectiveProblem
from geneticengine.representations.api import Representation


class SynthesisAlgorithm(ABC):
    tracker: ProgressTracker
    problem: Problem
    budget: SearchBudget
    representation: Representation

    def __init__(
        self,
        problem: Problem,
        budget: SearchBudget,
        representation: Representation,
        tracker: Optional[ProgressTracker] = None,
    ):
        self.problem = problem
        self.budget = budget
        self.representation = representation

        if tracker is None:
            if isinstance(problem, SingleObjectiveProblem):
                self.tracker = SingleObjectiveProgressTracker(problem, SequentialEvaluator())
            else:
                self.tracker = MultiObjectiveProgressTracker(problem, SequentialEvaluator())

        else:
            self.tracker = tracker

    def is_done(self) -> bool:
        """Whether the synthesis should stop, or not."""
        return self.budget.is_done(self.tracker)

    def get_problem(self) -> Problem:
        return self.problem

    @abstractmethod
    def search(self): ...
