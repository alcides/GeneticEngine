from __future__ import annotations

from dataclasses import dataclass
import sys

from geneticengine.grammar.grammar import Grammar
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import (
    RepresentationWithCrossover,
    RepresentationWithMutation,
    Representation,
)
from geneticengine.representations.tree.initializations import SynthesisDecider
from geneticengine.representations.tree.treebased import random_node
from geneticengine.solutions.tree import TreeNode
from geneticengine.nxt.linear_mutation import LinearGenomeMutation, PointMutation


@dataclass
class Genotype:
    dna: list[int]


@dataclass
class ListWrapper(RandomSource):
    dna: list[int]
    index: int = 0

    def randint(self, min: int, max: int) -> int:
        self.index = (self.index + 1) % len(self.dna)
        v = self.dna[self.index]
        return v % (max - min + 1) + min

    def random_float(self, min: float, max: float) -> float:
        k = self.randint(1, sys.maxsize)
        return 1 * (max - min) / k + min


class GrammaticalEvolutionRepresentation(
    Representation[Genotype, TreeNode],
    RepresentationWithMutation[Genotype],
    RepresentationWithCrossover[Genotype],
):
    def __init__(
        self,
        grammar: Grammar,
        decider: SynthesisDecider,
        gene_length: int = 256,
        mutation: LinearGenomeMutation[int] | None = None,
    ):
        """
        Args:
            grammar (Grammar): The grammar to use in the mapping
            decider (SynthesisDecider): Controls phenotype tree construction depth
            gene_length (int): Initial genome length for new individuals
            mutation: Mutation strategy for the integer codon list. Defaults to
                :class:`~geneticengine.nxt.linear_mutation.PointMutation`. Pass
                :class:`~geneticengine.nxt.linear_mutation.UMAD` for
                addition/deletion mutation.
        """
        self.grammar = grammar
        self.decider = decider
        self.gene_length = gene_length
        self.mutation: LinearGenomeMutation[int] = mutation if mutation is not None else PointMutation()

    def create_genotype(self, random: RandomSource, **kwargs) -> Genotype:
        return Genotype([random.randint(0, sys.maxsize) for _ in range(self.gene_length)])

    def genotype_to_phenotype(self, genotype: Genotype) -> TreeNode:
        rand: RandomSource = ListWrapper(genotype.dna)
        return random_node(rand, self.grammar, self.grammar.starting_symbol, self.decider)

    def _random_gene(self, random: RandomSource) -> int:
        return random.randint(0, sys.maxsize)

    def mutate(self, random: RandomSource, genotype: Genotype, **kwargs) -> Genotype:
        dna = self.mutation.mutate(
            list(genotype.dna),
            random,
            gene_factory=lambda: self._random_gene(random),
        )
        return Genotype(dna)

    def crossover(
        self,
        random: RandomSource,
        parent1: Genotype,
        parent2: Genotype,
        **kwargs,
    ) -> tuple[Genotype, Genotype]:
        limit = min(len(parent1.dna), len(parent2.dna))
        if limit <= 0:
            return (Genotype(list(parent1.dna)), Genotype(list(parent2.dna)))
        rindex = random.randint(0, limit - 1)
        c1 = parent1.dna[:rindex] + parent2.dna[rindex:]
        c2 = parent2.dna[:rindex] + parent1.dna[rindex:]
        return (Genotype(c1), Genotype(c2))
