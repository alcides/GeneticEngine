from abc import ABC, abstractmethod

from geneticengine.evaluation.tracker import (
    MultiObjectiveProgressTracker,
    ProgressTracker,
    SingleObjectiveProgressTracker,
)


class SearchBudget(ABC):
    @abstractmethod
    def is_done(self, tracker: ProgressTracker):
        ...


class TimeBudget(SearchBudget):
    def __init__(self, time: int):
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


class TargetFitness(SearchBudget):
    def __init__(self, value: float):
        self.value = float(value)

    def is_done(self, tracker: ProgressTracker):
        assert isinstance(tracker, SingleObjectiveProgressTracker)
        comps = tracker.get_best_individual().get_fitness(tracker.get_problem()).fitness_components
        if isinstance(self.value, float):
            return abs(comps[0] - self.value) < 0.0001
        else:
            return all(abs(c - self.value) < 0.001 for c in comps)


class TargetMultiFitness(SearchBudget):
    def __init__(self, targets: list[float]):
        self.targets = targets

    def is_done(self, tracker: ProgressTracker):
        assert isinstance(tracker, MultiObjectiveProgressTracker)
        comps = tracker.get_best_individuals()[0].get_fitness(tracker.get_problem()).fitness_components
        return all(abs(c - v) < 0.001 for v, c in zip(self.targets, comps))
