from abc import ABCMeta
from pickle import _Pickler as StockPickler
from typing import Any  # attr-defined: ignore
from dill import register
from geneticengine.solutions.individual import Individual
from geneticengine.problems import Fitness, Problem
from geneticengine.evaluation.api import Evaluator
from pathos.multiprocessing import ProcessingPool as Pool  # pyright: ignore


@register(ABCMeta)
def save_abc(pickler, obj):
    StockPickler.save_type(pickler, obj)


class ParallelEvaluator(Evaluator):
    """Evaluates individuals in parallel, each time they are needed."""

    def evaluate_async(self, problem: Problem, indivs: list[Individual[Any, Any]]):
        def mapper(ind: Individual) -> Fitness:
            return self.eval_single(problem, ind)

        with Pool(len(indivs)) as pool:
            fitnesses = pool.map(mapper, indivs)
            for i, f in zip(indivs, fitnesses):
                i.set_fitness(problem, f)
                self.register_evaluation()
                yield i
