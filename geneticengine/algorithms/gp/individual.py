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

    def evaluate(self, problem: Problem) -> FitnessType:
        if self.fitness is None:
            self.fitness = problem.evaluate(self.get_phenotype())
        return self.fitness

    def count_prods(self, all_nodes):
        counts = {prod: 1 for prod in all_nodes}

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

        counting(self.get_phenotype())
        # self.counts = counts
        return counts

    def production_probabilities(self, g):
        counts = self.count_prods(g.all_nodes)
        probs = counts.copy()
        for rule in g.alternatives:
            prods = g.alternatives[rule]
            total_counts = 0
            for prod in prods:
                total_counts += counts[prod]
            for prod in prods:
                probs[prod] = counts[prod] / total_counts

        for prod in probs.keys():
            if probs[prod] > 1:
                probs[prod] = 1

        return probs
