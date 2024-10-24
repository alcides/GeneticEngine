from abc import ABC
from time import monotonic_ns
from typing import Iterable
from geneticengine.evaluation import Evaluator
from geneticengine.evaluation.recorder import SearchRecorder
from geneticengine.evaluation.sequential import SequentialEvaluator
from geneticengine.problems import MultiObjectiveProblem, Problem, SingleObjectiveProblem


from geneticengine.solutions import Individual


class IndividualMemory(ABC):
    def append(self, ind: Individual) -> bool: ...
    def to_list(self) -> list[Individual]: ...


class PriorityQueue(IndividualMemory):
    def __init__(self, problem: Problem):
        assert isinstance(problem, SingleObjectiveProblem)
        self.problem = problem
        self.best = None

    def append(self, ind) -> bool:
        if self.best is None:
            self.best = ind
            return True
        elif self.problem.is_better(ind.get_fitness(self.problem), self.best.get_fitness(self.problem)):
            self.best = ind
            return True
        else:
            return False

    def to_list(self):
        return [self.best]


class ParetoFront(IndividualMemory):

    front: list[Individual]

    def __init__(self, problem: Problem):
        assert isinstance(problem, MultiObjectiveProblem)
        self.front = []
        self.problem = problem

    def is_dominated(self, current: Individual, others: list[Individual]):
        return all(
            [self.problem.is_better(x.get_fitness(self.problem), current.get_fitness(self.problem)) for x in others],
        )

    def append(self, ind) -> bool:
        if not self.front:
            self.front = [ind]
        elif self.is_dominated(ind, self.front):
            pass
        else:
            new_pareto_front = [ind]
            for old in self.front:
                if not self.is_dominated(old, new_pareto_front):
                    new_pareto_front.append(old)
            self.front = new_pareto_front
        return ind in self.front

    def to_list(self):
        return self.front


class ProgressTracker:
    problem: Problem
    evaluator: Evaluator
    start_time: int
    recorders: list[SearchRecorder]
    memory: IndividualMemory

    def __init__(self, problem: Problem, evaluator: Evaluator = None, recorders: list[SearchRecorder] = None):
        self.start_time = monotonic_ns()
        self.problem = problem
        self.evaluator = evaluator if evaluator is not None else SequentialEvaluator()
        self.recorders = recorders if recorders is not None else []
        self.memory = PriorityQueue(problem) if isinstance(problem, SingleObjectiveProblem) else ParetoFront(problem)

    def get_problem(self) -> Problem:
        return self.problem

    def get_number_evaluations(self) -> int:
        """The cumulative number of evaluations performed."""
        return self.evaluator.number_of_evaluations()

    def get_elapsed_time(self) -> float:
        """The elapsed time since the start in seconds."""
        return (monotonic_ns() - self.start_time) * 0.000000001  # seconds

    def evaluate(self, individuals: Iterable[Individual]):
        for ind in self.evaluator.evaluate_async(self.problem, individuals):
            is_best = self.memory.append(ind)
            for recorder in self.recorders:
                recorder.register(tracker=self, individual=ind, problem=self.problem, is_best=is_best)

    def evaluate_single(self, individual: Individual):
        self.evaluate([individual])

    def get_best_individuals(self) -> list[Individual]:
        return self.memory.to_list()
