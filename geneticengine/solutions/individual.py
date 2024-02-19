from __future__ import annotations

from typing import Any
from typing import Generic
from typing import TypeVar
import weakref

from geneticengine.problems import Fitness, Problem
from geneticengine.representations.api import Representation

G = TypeVar("G")
P = TypeVar("P")


class IndividualNotEvaluatedException(Exception):
    pass


class Individual(Generic[G, P]):
    genotype: G
    representation: Representation[G, P]
    phenotype: P | None = None
    fitness_store: weakref.WeakKeyDictionary[Problem, Fitness]
    metadata: dict[str, Any]

    def __init__(self, genotype: G, representation: Representation[G, P], metadata: dict[str, Any] = None):
        self.genotype = genotype
        self.representation = representation
        self.fitness_store: weakref.WeakKeyDictionary[Problem, Fitness] = weakref.WeakKeyDictionary()
        self.metadata = {} if metadata is None else metadata

    def get_phenotype(self):
        if self.phenotype is None:
            self.phenotype = self.representation.genotype_to_phenotype(self.genotype)
        return self.phenotype

    def has_fitness(self, problem: Problem) -> bool:
        return problem in self.fitness_store

    def set_fitness(self, problem: Problem, fitness: Fitness):
        self.fitness_store[problem] = fitness

    def get_fitness(self, problem: Problem | None = None) -> Fitness:
        if problem is None:
            default_problem = list(self.fitness_store.keys())[0]
            return self.fitness_store[default_problem]
        elif problem in self.fitness_store:
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
