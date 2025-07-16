from __future__ import annotations

from typing import Any, Iterator, Sequence

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
import numpy as np

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

        counting(individual)
        return counts

    def compute_production_probabilities(self, individuals: Sequence[Individual], g: Grammar):
        """Calculates production probabilities by aggregating counts from a list of individuals."""
        total_counts = {prod: 0 for prod in g.all_nodes}
        for individual in individuals:
            individual_counts = self.count_productions(individual.get_phenotype(), g)
            for prod, count in individual_counts.items():
                total_counts[prod] += count

        probs: dict[type, float] = {prod: 0.0 for prod in g.all_nodes}
        for rule in g.alternatives:
            prods = g.alternatives[rule]
            total_rule_counts = sum(total_counts.get(p, 0) for p in prods)

            for prod in prods:
                probs[prod] = total_counts.get(prod, 0) / total_rule_counts if total_rule_counts > 0 else 0.0

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
        candidates = list(evaluator.evaluate(problem, population_list))

        pareto_front = list(non_dominated(iter(candidates), problem))

        if not pareto_front:
            return iter(candidates)

        assert isinstance(representation, TreeBasedRepresentation)
        probs = self.compute_production_probabilities(pareto_front, representation.grammar)
        representation.grammar = representation.grammar.update_weights(self.learning_rate, probs)

        return iter(candidates)

class ConditionalWeightLearningStep(GeneticStep):
    """
    A genetic step that applies weight learning using only individuals
    from the Pareto front that are above a specified fitness threshold.
    """

    def __init__(
        self,
        fitness_threshold: float = 0.5,
        weight_learning_rate: float = 0.01,
    ):
        """
        Args:
            fitness_threshold (float): The minimum average fitness an individual
                must have to be included in the weight learning process.
            weight_learning_rate (float): The learning rate for the grammar update.
        """
        self.fitness_threshold = fitness_threshold
        self.internal_weight_learning_step = WeightLearningStep(
            learning_rate=weight_learning_rate,
        )

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
        """
        The main iteration logic for the conditional step.
        """
        population_list = list(population)
        candidates = list(evaluator.evaluate(problem, population_list))
        pareto_front = list(non_dominated(iter(candidates), problem))

        if not pareto_front:
            yield from candidates
            return

        individuals_for_learning = [
            ind for ind in pareto_front
            if np.mean(ind.get_fitness(problem).fitness_components) >= self.fitness_threshold
        ]

        if individuals_for_learning:
            assert isinstance(representation, TreeBasedRepresentation)

            probs = self.internal_weight_learning_step.compute_production_probabilities(
                individuals_for_learning, representation.grammar,
            )

            representation.grammar = representation.grammar.update_weights(
                self.internal_weight_learning_step.learning_rate, probs,
            )

        yield from candidates
