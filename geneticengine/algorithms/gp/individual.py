from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from geneticengine.core.problems import FitnessType


@dataclass
class Individual:
    genotype: Any
    fitness: FitnessType | None = None

    def __str__(self) -> str:
        return str(self.genotype)

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

        counting(genotype_to_phenotype(g, self.genotype))
        # self.counts = counts
        return counts

    def production_probabilities(self, genotype_to_phenotype, g):
        counts = self.count_prods(genotype_to_phenotype, g)
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
