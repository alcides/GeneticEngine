from abc import ABC
from time import time_ns
from typing import Optional
from geneticengine.evaluation import Evaluator
from geneticengine.evaluation.recorder import SearchRecorder
from geneticengine.evaluation.sequential import SequentialEvaluator
from geneticengine.problems import Problem
from geneticengine.solutions import Individual


class ProgressTracker(ABC):
    problem: Problem
    evaluator: Evaluator
    start_time: int
    recorders: list[SearchRecorder]

    def __init__(self, problem: Problem, evaluator: Evaluator, recorders: list[SearchRecorder] = None):
        self.start_time = time_ns()
        self.problem = problem
        self.evaluator = evaluator if evaluator is not None else SequentialEvaluator()
        self.recorders = recorders if recorders is not None else []


class SingleObjectiveProgressTracker(ProgressTracker):

    best_individual: Optional[Individual]

    def __init__(
        self,
        problem: Problem,
        evaluator: Evaluator = None,
        recorders: list[SearchRecorder] = None,
    ):
        super().__init__(problem, evaluator, recorders=recorders)

        self.best_individual = None

    def evaluate(self, individuals: list[Individual]):
        problem = self.problem
        self.evaluator.evaluate(problem, individuals)
        for ind in individuals:
            is_best = False
            if self.best_individual is None:
                self.best_individual = ind
                is_best = True
            elif problem.is_better(ind.get_fitness(problem), self.best_individual.get_fitness(problem)):
                self.best_individual = ind
                is_best = True
            else:
                is_best = False
            for recorder in self.recorders:
                recorder.register(tracker=self, individual=ind, problem=problem, is_best=is_best)

    def get_best_individual(self) -> Individual:
        return self.best_individual

    def get_elapsed_time(self) -> float:
        """The elapsed time since the start in seconds."""
        return (time_ns() - self.start_time) / 1_000_000  # seconds

    def get_number_evaluations(self) -> int:
        """The cumulative number of evaluations performed."""
        return self.evaluator.number_of_evaluations()

    def get_problem(self) -> Problem:
        return self.problem
