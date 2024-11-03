from abc import ABC, abstractmethod

from geneticengine.evaluation.tracker import (
    ProgressTracker,
)


class SearchBudget(ABC):
    @abstractmethod
    def is_done(self, tracker: ProgressTracker): ...


class TimeBudget(SearchBudget):
    def __init__(self, time: float):
        self.time_budget = time

    def is_done(self, tracker: ProgressTracker):
        return tracker.get_elapsed_time() >= self.time_budget


class EvaluationBudget(SearchBudget):
    def __init__(self, evaluations: int):
        self.evaluations_budget = evaluations

    def is_done(self, tracker: ProgressTracker):
        return tracker.get_number_evaluations() >= self.evaluations_budget


class AnyOf(SearchBudget):
    def __init__(self, a: SearchBudget, b: SearchBudget):
        self.a = a
        self.b = b

    def is_done(self, tracker: ProgressTracker):
        return self.a.is_done(tracker) or self.b.is_done(tracker)
