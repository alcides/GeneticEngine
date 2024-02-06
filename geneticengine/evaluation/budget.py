from abc import ABC, abstractmethod

from geneticengine.evaluation.recorder import SingleObjectiveProgressTracker


class SearchBudget(ABC):
    @abstractmethod
    def is_done(self, tracker: SingleObjectiveProgressTracker):
        ...


class TimeBudget(SearchBudget):
    def __init__(self, time: int):
        self.time_budget = time

    def is_done(self, tracker: SingleObjectiveProgressTracker):
        return tracker.get_elapsed_time() >= self.time_budget


class EvaluationBudget(SearchBudget):
    def __init__(self, evaluations: int):
        self.evaluations_budget = evaluations

    def is_done(self, tracker: SingleObjectiveProgressTracker):
        return tracker.get_number_evaluations() >= self.evaluations_budget
