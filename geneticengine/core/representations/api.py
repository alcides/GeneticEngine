from __future__ import annotations

import abc
from typing import Generic
from typing import TypeVar

from geneticengine.core.grammar import Grammar
from geneticengine.core.problems import Problem
from geneticengine.core.random.sources import Source
from geneticengine.core.evaluators import Evaluator

g = TypeVar("g")
p = TypeVar("p")


class MutationOperator(Generic[g], abc.ABC):
    """This class wraps possible mutation operators, which can be used with
    GenericMutationStep."""

    @abc.abstractmethod
    def mutate(
        self,
        genotype: g,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random_source: Source,
        index_in_population: int,
        generation: int,
    ) -> g:
        ...


class CrossoverOperator(Generic[g], abc.ABC):
    """This class wraps possible mutation operators, which can be used with
    GenericCrossoverStep."""

    @abc.abstractmethod
    def crossover(
        self,
        g1: g,
        g2: g,
        problem: Problem,
        representation: Representation,
        random_source: Source,
        index_in_population: int,
        generation: int,
    ) -> tuple[g, g]:
        ...


class Representation(Generic[g, p]):
    grammar: Grammar
    min_depth: int
    max_depth: int

    def __init__(self, grammar: Grammar, max_depth: int):
        self.grammar = grammar
        self.min_depth = self.grammar.get_min_tree_depth()
        self.max_depth = max_depth  # Old version: min(max_depth, self.grammar.get_max_node_depth())
        assert self.min_depth <= self.max_depth

    @abc.abstractmethod
    def create_individual(self, r: Source, depth: int | None = None, **kwargs) -> g:
        ...

    @abc.abstractmethod
    def get_mutation(self) -> MutationOperator[g]:
        ...

    @abc.abstractmethod
    def get_crossover(self) -> CrossoverOperator[g]:
        ...

    @abc.abstractmethod
    def genotype_to_phenotype(self, genotype: g) -> p:
        ...

    @abc.abstractmethod
    def phenotype_to_genotype(self, phenotype: p) -> g:
        """Takes an existing program and adapts it to be used in the right
        representation."""
        ...
