from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from os import cpu_count
from typing import Any, Generator, Iterable

from geneticengine.problems import Fitness, InvalidFitnessException, Problem
from geneticengine.evaluation.api import Evaluator, IndT


class ParallelEvaluator(Evaluator):
    """Evaluates individuals lazily in bounded batches of worker threads."""

    def evaluate_async(
        self,
        problem: Problem,
        individuals: Iterable[IndT],
    ) -> Generator[IndT, Any, Any]:
        def mapper(ind: IndT) -> tuple[IndT, Fitness | None]:
            try:
                return ind, self.eval_single(problem, ind)
            except InvalidFitnessException:
                return ind, None

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            while batch := list(islice(individuals, self.workers)):
                fitnesses = list(executor.map(mapper, batch))

                for i, f in fitnesses:
                    if f is None:
                        self.register_invalid_evaluation()
                        continue
                    i.set_fitness(problem, f)
                    self.register_evaluation(i, problem)
                    yield i

    def __init__(self, workers: int | None = None):
        super().__init__()
        self.workers = workers or min(cpu_count() or 1, 8)
