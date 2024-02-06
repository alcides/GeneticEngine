from abc import ABC, abstractmethod
from typing import Any, Generator


from geneticengine.solutions.individual import Individual
from geneticengine.problems import Problem


class Evaluator(ABC):
    def __init__(self):
        self.count = 0

    @abstractmethod
    def eval(self, problem: Problem, indivs: list[Individual[Any, Any]]) -> Generator[Individual, None, None]:
        ...

    def register_evaluation(self):
        self.count += 1

    def get_count(self):
        return self.count

    def eval_single(self, problem: Problem, individual: Individual):
        if not individual.has_fitness(problem):
            phenotype = individual.get_phenotype()
            individual.set_fitness(problem, problem.evaluate(phenotype=phenotype))
            self.register_evaluation()
            yield individual
