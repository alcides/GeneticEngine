from __future__ import annotations

from typing import Any, Iterator

from geneticengine.solutions.individual import PhenotypicIndividual, Individual
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.problems import Problem
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import Representation
from geneticengine.evaluation import Evaluator
from geneticengine.grammar import Grammar
from geneticengine.solutions.tree import TreeNode
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.problems.helpers import non_dominated

class WeightLearningStep(GeneticStep):
    """Applies weight learning to the grammar with a given learning rate."""

    def __init__(
        self,
        learning_rate: float = 0.01,
    ):
        self.learning_rate = learning_rate

    def wrap(self, representation: Representation, genotype: Any) -> PhenotypicIndividual:
        return PhenotypicIndividual(
            genotype=genotype,
            representation=representation,
        )

    def count_productions(self, individual: TreeNode, g: Grammar):
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
        return counts

    def production_probabilities(self, individual: Individual, g: Grammar):
        counts = self.count_productions(individual.get_phenotype(), g)
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

    def iterate(
        self,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random: RandomSource,
        population: Iterator[PhenotypicIndividual],
        target_size: int,
        generation: int,
    ) -> Iterator[PhenotypicIndividual]:
        population_list = list(population)
        candidates = evaluator.evaluate(problem, iter(population_list))
        best = next(non_dominated(iter(candidates), problem))
        assert isinstance(representation, TreeBasedRepresentation)
        probs = self.production_probabilities(best, representation.grammar)
        representation.grammar = representation.grammar.update_weights(self.learning_rate, probs)
        for index, ind in enumerate(population_list):
            if index < target_size:
                nind = self.wrap(representation, ind.genotype)
                yield nind

