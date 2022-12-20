from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from typing import Generic
from typing import TypeVar

from geneticengine.core.problems import FitnessType
from geneticengine.core.problems import Problem

G = TypeVar("G")
P = TypeVar("P")


@dataclass
class Individual(Generic[G, P]):
    genotype: G
    genotype_to_phenotype: Callable[[G], P]
    phenotype: P | None = None
    fitness: FitnessType | None = None

    def __str__(self) -> str:
        return f"{self.genotype}"

    def get_phenotype(self):
        if self.phenotype is None:
            self.phenotype = self.genotype_to_phenotype(self.genotype)
        return self.phenotype

    def evaluate(self, problem: Problem) -> FitnessType:
        if self.fitness is None:
            self.fitness = problem.evaluate(self.get_phenotype())
        return self.fitness
