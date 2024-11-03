from abc import ABCMeta
from pickle import _Pickler as StockPickler
from typing import Any, Generator, Iterable  # attr-defined: ignore
from dill import register
from geneticengine.solutions.individual import Individual
from geneticengine.problems import Fitness, Problem
from geneticengine.evaluation.api import Evaluator


@register(ABCMeta)
def save_abc(pickler, obj):
    StockPickler.save_type(pickler, obj)  # pyright: ignore


class ParallelEvaluator(Evaluator):
    """Evaluates individuals in parallel, each time they are needed."""

    def evaluate_async(
        self,
        problem: Problem,
        individuals: Iterable[Individual],
    ) -> Generator[Individual, Any, Any]:
        indivs = list(individuals)

        def mapper(ind: Individual) -> Fitness:
            return self.eval_single(problem, ind)

        from pathos.multiprocessing import ProcessingPool as Pool  # pyright: ignore

        with Pool(len(indivs)) as pool:
            fitnesses = pool.map(mapper, indivs)
            for i, f in zip(indivs, fitnesses):
                i.set_fitness(problem, f)
                self.register_evaluation(i, problem)
                yield i
