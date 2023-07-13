from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from geneticengine.core.grammar import Grammar
from geneticengine.core.problems import Problem
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import CrossoverOperator
from geneticengine.core.representations.api import MutationOperator
from geneticengine.core.representations.api import Representation
from geneticengine.core.representations.tree.initializations import (
    InitializationMethodType,
)
from geneticengine.core.representations.tree.initializations import pi_grow_method
from geneticengine.core.representations.tree.treebased import random_node
from geneticengine.core.tree import TreeNode
from geneticengine.core.evaluators import Evaluator

MAX_VALUE = 10000000
GENE_LENGTH = 256


@dataclass
class Genotype:
    dna: list[int]


def random_individual(
    r: Source,
    g: Grammar,
    depth: int = 5,
    starting_symbol: Any = None,
) -> Genotype:
    return Genotype([r.randint(0, MAX_VALUE) for _ in range(GENE_LENGTH)])


def mutate(r: Source, g: Grammar, ind: Genotype, max_depth: int) -> Genotype:
    rindex = r.randint(0, 255)
    clone = [i for i in ind.dna]
    clone[rindex] = r.randint(0, 10000)
    return Genotype(clone)


def crossover(
    r: Source,
    g: Grammar,
    p1: Genotype,
    p2: Genotype,
    max_depth: int,
) -> tuple[Genotype, Genotype]:
    rindex = r.randint(0, 255)
    c1 = p1.dna[:rindex] + p2.dna[rindex:]
    c2 = p2.dna[:rindex] + p1.dna[rindex:]
    return (Genotype(c1), Genotype(c2))


@dataclass
class ListWrapper(Source):
    dna: list[int]
    index: int = 0

    def randint(self, min: int, max: int, prod: str = "") -> int:
        self.index = (self.index + 1) % len(self.dna)
        v = self.dna[self.index]
        return v % (max - min + 1) + min

    def random_float(self, min: float, max: float, prod: str = "") -> float:
        k = self.randint(1, MAX_VALUE, prod)
        return 1 * (max - min) / k + min


def create_tree(
    g: Grammar,
    ind: Genotype,
    depth: int,
    initialization_mode: InitializationMethodType = pi_grow_method,
) -> TreeNode:
    rand: Source = ListWrapper(ind.dna)
    return random_node(rand, g, depth, g.starting_symbol, initialization_mode)


class DefaultGEMutation(MutationOperator[Genotype]):
    """Chooses a position in the list, and mutates it."""

    def mutate(
        self,
        genotype: Genotype,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random_source: Source,
        index_in_population: int,
        generation: int,
    ) -> Genotype:
        return mutate(
            random_source,
            representation.grammar,
            genotype,
            representation.max_depth,
        )


class DefaultGECrossover(CrossoverOperator[Genotype]):
    """One-point crossover between the lists."""

    def crossover(
        self,
        g1: Genotype,
        g2: Genotype,
        problem: Problem,
        representation: Representation,
        random_source: Source,
        index_in_population: int,
        generation: int,
    ) -> tuple[Genotype, Genotype]:
        return crossover(random_source, representation.grammar, g1, g2, representation.max_depth)


class GrammaticalEvolutionRepresentation(Representation[Genotype, TreeNode]):
    """This representation uses a list of integers to guide the generation of
    trees in the phenotype."""

    def __init__(
        self,
        grammar: Grammar,
        max_depth: int,
        initialization_mode: InitializationMethodType = pi_grow_method,
    ):
        """
        Args:
            grammar (Grammar): The grammar to use in the mapping
            max_depth (int): the maximum depth when performing the mapping
            initialization_mode (InitializationMethodType): method to create individuals in the mapping
                (e.g., pi_grow, full, grow)
        """
        super().__init__(grammar, max_depth)
        self.initialization_mode = initialization_mode

    def create_individual(
        self,
        r: Source,
        depth: int | None = None,
        **kwargs,
    ) -> Genotype:
        actual_depth = depth or self.max_depth
        return random_individual(r, self.grammar, depth=actual_depth)

    def genotype_to_phenotype(self, genotype: Genotype) -> TreeNode:
        return create_tree(
            self.grammar,
            genotype,
            self.max_depth,
            self.initialization_mode,
        )

    def phenotype_to_genotype(self, phenotype: Any) -> Genotype:
        """Takes an existing program and adapts it to be used in the right
        representation."""
        raise NotImplementedError(
            "Reconstruction of genotype not supported in this representation.",
        )

    def get_mutation(self) -> MutationOperator[Genotype]:
        return DefaultGEMutation()

    def get_crossover(self) -> CrossoverOperator[Genotype]:
        return DefaultGECrossover()
