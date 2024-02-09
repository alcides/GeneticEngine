from __future__ import annotations

from dataclasses import dataclass
import sys

from geneticengine.grammar.grammar import Grammar
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import (
    RepresentationWithCrossover,
    RepresentationWithMutation,
    SolutionRepresentation,
)
from geneticengine.representations.tree.initializations import (
    InitializationMethodType,
)
from geneticengine.representations.tree.initializations import pi_grow_method
from geneticengine.representations.tree.treebased import random_node
from geneticengine.solutions.tree import TreeNode


@dataclass
class Genotype:
    dna: list[int]


@dataclass
class ListWrapper(RandomSource):
    dna: list[int]
    index: int = 0

    def randint(self, min: int, max: int, prod: str = "") -> int:
        self.index = (self.index + 1) % len(self.dna)
        v = self.dna[self.index]
        return v % (max - min + 1) + min

    def random_float(self, min: float, max: float, prod: str = "") -> float:
        k = self.randint(1, sys.maxsize, prod)
        return 1 * (max - min) / k + min


class GrammaticalEvolutionRepresentation(
    SolutionRepresentation[Genotype, TreeNode],
    RepresentationWithMutation[Genotype],
    RepresentationWithCrossover[Genotype],
):

    def __init__(
        self,
        grammar: Grammar,
        max_depth: int,  # TODO: parameterize
        gene_length: int = 256,
        initialization_mode: InitializationMethodType = pi_grow_method,
    ):
        """
        Args:
            grammar (Grammar): The grammar to use in the mapping
            max_depth (int): the maximum depth when performing the mapping
            initialization_mode (InitializationMethodType): method to create individuals in the mapping
                (e.g., pi_grow, full, grow)
        """
        self.grammar = grammar
        self.max_depth = max_depth
        self.gene_length = gene_length
        self.initialization_mode = initialization_mode

    def instantiate(self, random: RandomSource, **kwargs) -> Genotype:
        return Genotype([random.randint(0, sys.maxsize) for _ in range(self.gene_length)])

    def map(self, genotype: Genotype) -> TreeNode:
        rand: RandomSource = ListWrapper(genotype.dna)
        return random_node(rand, self.grammar, self.max_depth, self.grammar.starting_symbol, self.initialization_mode)

    def mutate(self, random: RandomSource, internal: Genotype, **kwargs) -> Genotype:
        rindex = random.randint(0, self.gene_length - 1)
        clone = [i for i in internal.dna]
        clone[rindex] = random.randint(0, sys.maxsize)
        return Genotype(clone)

    def crossover(
        self, random: RandomSource, parent1: Genotype, parent2: Genotype, **kwargs
    ) -> tuple[Genotype, Genotype]:
        rindex = random.randint(0, self.gene_length - 1)
        c1 = parent1.dna[:rindex] + parent2.dna[rindex:]
        c2 = parent2.dna[:rindex] + parent1.dna[rindex:]
        return (Genotype(c1), Genotype(c2))
