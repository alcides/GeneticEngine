from abc import ABC, abstractmethod
import logging
from typing import Any, Generator, Iterable, TypeVar


from geneticengine.evaluation.exceptions import IndividualFoundException
from geneticengine.solutions.individual import Individual
from geneticengine.problems import Fitness, Problem

logger = logging.getLogger(__name__)

IndT = TypeVar("IndT", bound=Individual)

class Evaluator(ABC):
    def __init__(self):
        self.count = 0

    @abstractmethod
    def evaluate_async(
        self,
        problem: Problem,
        individuals: Iterable[IndT],
    ) -> Generator[IndT, Any, Any]: ...

    def evaluate(self, problem: Problem, individuals: Iterable[IndT]) -> Iterable[IndT]:
        yield from self.evaluate_async(problem, individuals)

    def register_evaluation(self, individual: IndT, problem: Problem):
        self.count += 1
        if problem.is_solved(individual.get_fitness(problem)):
            raise IndividualFoundException(individual)

    def number_of_evaluations(self):
        return self.count

    def eval_single(self, problem: Problem, individual: IndT) -> Fitness:
        phenotype = individual.get_phenotype()
        r = problem.evaluate(phenotype=phenotype)
        logger.debug(f"Evaluated #{id(phenotype)}: {r}")
        return r
