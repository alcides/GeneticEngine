from abc import ABC, abstractmethod
import logging
from typing import Any, Generator, Iterable


from geneticengine.solutions.individual import Individual
from geneticengine.problems import Fitness, Problem

logger = logging.getLogger(__name__)


class Evaluator(ABC):
    def __init__(self):
        self.count = 0

    @abstractmethod
    def evaluate_async(self, problem: Problem, individuals: Iterable[Individual[Any, Any]]) -> Generator[Individual, Any, Any]: ...

    def evaluate(self, problem: Problem, individuals: Iterable[Individual[Any, Any]]):
        for _ in self.evaluate_async(problem, individuals):
            pass

    def register_evaluation(self):
        self.count += 1

    def number_of_evaluations(self):
        return self.count

    def eval_single(self, problem: Problem, individual: Individual) -> Fitness:
        phenotype = individual.get_phenotype()
        r = problem.evaluate(phenotype=phenotype)
        logger.debug(f"Evaluated #{id(phenotype)}: {r}")
        return r
