from __future__ import annotations

from typing import Any

from geneticengine.algorithms.gp.individual import Individual
from geneticengine.core.grammar import Grammar
from geneticengine.core.tree import TreeNode


def count_productions(individual: TreeNode, g: Grammar):
    """
    Returns the number of occurences of each non terminal.
    """
    counts = {prod: 1 for prod in g.all_nodes}

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

    counting(individual)
    # self.counts = counts
    return counts


def production_probabilities(individual: Individual, g: Grammar):
    """
    Returns the probability of seeing a certain production occur in an individual.
    """
    counts = count_productions(individual.get_phenotype(), g)
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
