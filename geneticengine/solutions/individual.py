from __future__ import annotations

from abc import ABC, abstractmethod
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


class Individual(Generic[P], ABC):
    fitness_store: weakref.WeakKeyDictionary[Problem, Fitness]
    metadata: dict[str, Any]

    def __init__(self, metadata: dict[str, Any] | None = None):
        self.fitness_store: weakref.WeakKeyDictionary[Problem, Fitness] = weakref.WeakKeyDictionary()
        self.metadata = {} if metadata is None else metadata

    @abstractmethod
    def get_phenotype(self) -> P: ...

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


class ConcreteIndividual(Generic[P], Individual[P]):
    instance: P

    def __init__(self, instance: P, metadata: dict[str, Any] | None = None):
        super().__init__(metadata=metadata)
        self.instance = instance

    def get_phenotype(self):
        return self.instance


class PhenotypicIndividual(Generic[G, P], Individual[P]):
    genotype: G
    representation: Representation[G, P]
    phenotype: P | None = None

    def __init__(self, genotype: G, representation: Representation[G, P], metadata: dict[str, Any] | None = None):
        super().__init__(metadata=metadata)
        self.genotype = genotype
        self.representation = representation

    def get_phenotype(self):
        if self.phenotype is None:
            self.phenotype = self.representation.genotype_to_phenotype(self.genotype)
        return self.phenotype

    def __str__(self) -> str:
        return f"{self.genotype}"
