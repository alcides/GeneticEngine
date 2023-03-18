from __future__ import annotations

from typing import Callable
from typing import Generic
from typing import TypeVar
import weakref
from geneticengine.core.generic_utils import GenericWrapper

from geneticengine.core.problems import Fitness, Problem

G = TypeVar("G")
P = TypeVar("P")


class IndividualNotEvaluatedException(Exception):
    pass


class Individual(Generic[G, P]):
    genotype: G
    genotype_to_phenotype: GenericWrapper[Callable[[G], P]]
    phenotype: P | None = None

    def __init__(self, genotype: G, genotype_to_phenotype: Callable[[G], P]):
        self.genotype = genotype
        self.genotype_to_phenotype = GenericWrapper(genotype_to_phenotype)
        self.fitness_store: weakref.WeakKeyDictionary[Problem, Fitness] = weakref.WeakKeyDictionary()

    def get_phenotype(self):
        if self.phenotype is None:
            self.phenotype = self.genotype_to_phenotype.get()(self.genotype)
        return self.phenotype

    def has_fitness(self, problem: Problem) -> bool:
        return problem in self.fitness_store

    def set_fitness(self, problem: Problem, fitness: Fitness):
        self.fitness_store[problem] = fitness

    def get_fitness(self, problem: Problem) -> Fitness:
        if problem in self.fitness_store:
            return self.fitness_store[problem]
        else:
            raise IndividualNotEvaluatedException()

    def ensure_fitness(self, problem: Problem):
        if not self.has_fitness(problem):
            self.set_fitness(problem, problem.evaluate(self.get_phenotype()))

    @staticmethod
    def key_function(problem: Problem):
        def kf(ind):
            ind.ensure_fitness(problem)
            return ind.get_fitness(problem)[0]

        return kf

    def __str__(self) -> str:
        return f"{self.genotype}"
