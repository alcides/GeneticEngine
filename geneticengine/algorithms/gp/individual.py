from __future__ import annotations

from dataclasses import dataclass
from typing import Any
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
        return str(self.genotype)

    def get_phenotype(self):
        if self.phenotype is None:
            self.phenotype = self.genotype_to_phenotype(self.genotype)
        return self.phenotype

    def count_prods(self, genotype_to_phenotype, g):
        counts = {prod: 0 for prod in g.all_nodes}

        def add_count(ty):
            if ty in counts.keys():
                counts[ty] += 1

        def get_args(no):
            if hasattr(type(no), "__annotations__"):
                return type(no).__annotations__.keys()
            return []

        def counting(node: Any):
            add_count(type(node))
            for base in type(node).__bases__:
                add_count(base)
            for argn in get_args(node):
                counting(getattr(node, argn))

        counting(genotype_to_phenotype(self.genotype))
        return counts

    def evaluate(self, problem: Problem) -> FitnessType:
        if self.fitness is None:
            self.fitness = problem.evaluate(self.get_phenotype())
        return self.fitness
