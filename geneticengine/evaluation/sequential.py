from typing import Any, Iterable, Generator

from geneticengine.solutions.individual import Individual
from geneticengine.problems import Problem
from geneticengine.evaluation.api import Evaluator


class SequentialEvaluator(Evaluator):
    """Default evaluator for individuals, executes sequentially."""

    def evaluate_async(self, problem: Problem, individuals: Iterable[Individual[Any, Any]]) -> Generator[Individual, Any, Any]:
        for individual in individuals:
            if not individual.has_fitness(problem):
                f = self.eval_single(problem, individual)
                self.register_evaluation()
                individual.set_fitness(problem, f)
            yield individual
