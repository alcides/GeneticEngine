from abc import ABC, abstractmethod
from typing import Any


from geneticengine.solutions.individual import Individual
from geneticengine.problems import Problem


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

    def eval_single(self, problem: Problem, individual: Individual):
        if not individual.has_fitness(problem):
            phenotype = individual.get_phenotype()
            individual.set_fitness(problem, problem.evaluate(phenotype=phenotype))
            self.register_evaluation()
