from abc import ABC, abstractmethod
from typing import Any


from geneticengine.solutions.individual import Individual
from geneticengine.problems import Fitness, Problem


class Evaluator(ABC):
    def __init__(self):
        self.count = 0

    @abstractmethod
    def evaluate(self, problem: Problem, indivs: list[Individual[Any, Any]]):
        ...

    def register_evaluation(self):
        self.count += 1

    def number_of_evaluations(self):
        return self.count

    def eval_single(self, problem: Problem, individual: Individual) -> Fitness:
        phenotype = individual.get_phenotype()
        r = problem.evaluate(phenotype=phenotype)
        return r
