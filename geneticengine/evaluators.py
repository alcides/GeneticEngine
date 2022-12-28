from abc import ABC, abstractmethod
from typing import Any
from geneticengine.algorithms.gp.individual import Individual

from geneticengine.core.problems import Problem


class Evaluator(ABC):
    @abstractmethod
    def eval(self, p: Problem, indivs: list[Individual[Any, Any]]):
        ...


class SequentialEvaluator(Evaluator):
    def eval(self, p: Problem, indivs: list[Individual[Any, Any]]):
        for individual in indivs:
            if not individual.has_fitness(p):
                f = p.evaluate(individual.get_phenotype())
                individual.set_fitness(p, f)
