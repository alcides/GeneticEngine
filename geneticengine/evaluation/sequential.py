from typing import Any, Generator

from geneticengine.solutions.individual import Individual
from geneticengine.problems import Problem
from geneticengine.evaluation import Evaluator


class SequentialEvaluator(Evaluator):
    """Default evaluator for individuals, executes sequentially."""

    def eval(self, problem: Problem, indivs: list[Individual[Any, Any]]) -> Generator[Individual, None, None]:
        for individual in indivs:
            self.eval_single(problem, individual)
        return None
