from typing import Any, Iterable, Generator

from geneticengine.problems import InvalidFitnessException, Problem
from geneticengine.evaluation.api import Evaluator, IndT


class SequentialEvaluator(Evaluator):
    """Default evaluator for individuals, executes sequentially."""

    def evaluate_async(
        self,
        problem: Problem,
        individuals: Iterable[IndT],
    ) -> Generator[IndT, Any, Any]:
        for individual in individuals:
            if not individual.has_fitness(problem):
                try:
                    f = self.eval_single(problem, individual)
                    individual.set_fitness(problem, f)
                    self.register_evaluation(individual, problem)
                    yield individual
                except InvalidFitnessException:
                    pass
            else:
                yield individual
