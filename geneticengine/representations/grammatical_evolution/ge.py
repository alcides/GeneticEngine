from __future__ import annotations

from dataclasses import dataclass
import sys

from geneticengine.grammar.grammar import Grammar
from geneticengine.random.sources import RandomSource
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.api import (
    RepresentationWithCrossover,
    RepresentationWithMutation,
    Representation,
)
from geneticengine.representations.tree.initializations import SynthesisDecider
from geneticengine.representations.tree.treebased import random_node
from geneticengine.solutions.tree import TreeNode
from geneticengine.representations.linear_mutation import LinearGenomeMutation, PointMutation


@dataclass
class Genotype:
    dna: list[int]


@dataclass
class ListWrapper(RandomSource):
    dna: list[int]
    index: int = 0
    random: RandomSource | None = None
    extend: bool = False
    consumed: int = 0

    def _next_gene(self) -> int:
        self.consumed += 1
        if self.index >= len(self.dna):
            if not self.extend or self.random is None:
                self.index = (self.index + 1) % len(self.dna)
            else:
                self.dna.append(self.random.randint(0, sys.maxsize))
        value = self.dna[self.index]
        self.index += 1
        return value

    def randint(self, min: int, max: int) -> int:
        v = self._next_gene()
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
        extend_genotype: bool = False,
    ):
        """
        Args:
            grammar (Grammar): The grammar to use in the mapping
            decider (SynthesisDecider): Controls phenotype tree construction depth
            gene_length (int): Initial genome length for new individuals
            mutation: Mutation strategy for the integer codon list. Defaults to
                :class:`~geneticengine.representations.linear_mutation.PointMutation`. Pass
                :class:`~geneticengine.representations.linear_mutation.UMAD` for
                addition/deletion mutation.
        """
        self.grammar = grammar
        self.decider = decider
        self.gene_length = gene_length
        self.mutation: LinearGenomeMutation[int] = mutation if mutation is not None else PointMutation()
        self.extend_genotype = extend_genotype

    def _expressed_length(self, genotype: Genotype) -> int:
        """Return the number of codons used to map ``genotype``."""
        rand = ListWrapper(
            list(genotype.dna),
            random=NativeRandomSource(0) if self.extend_genotype else None,
            extend=self.extend_genotype,
        )
        random_node(rand, self.grammar, self.grammar.starting_symbol, self.decider)
        return min(rand.consumed, len(genotype.dna))

    def create_genotype(self, random: RandomSource, **kwargs) -> Genotype:
        return Genotype([random.randint(0, sys.maxsize) for _ in range(self.gene_length)])

    def genotype_to_phenotype(self, genotype: Genotype) -> TreeNode:
        rand: RandomSource = ListWrapper(
            genotype.dna,
            random=NativeRandomSource(0) if self.extend_genotype else None,
            extend=self.extend_genotype,
        )
        return random_node(rand, self.grammar, self.grammar.starting_symbol, self.decider)

    def _random_gene(self, random: RandomSource) -> int:
        return random.randint(0, sys.maxsize)

    def mutate(self, random: RandomSource, genotype: Genotype, effective: bool = False, **kwargs) -> Genotype:
        active_length = self._expressed_length(genotype) if effective else None
        if effective and active_length == 0:
            return Genotype(list(genotype.dna))
        if effective and isinstance(self.mutation, PointMutation):
            dna = list(genotype.dna)
            index = random.randint(0, active_length - 1)
            dna[index] = self._random_gene(random)
            return Genotype(dna)
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
        effective: bool = False,
        **kwargs,
    ) -> tuple[Genotype, Genotype]:
        limit = min(len(parent1.dna), len(parent2.dna))
        if limit <= 0:
            return (Genotype(list(parent1.dna)), Genotype(list(parent2.dna)))
        if effective:
            limit = min(limit, self._expressed_length(parent1), self._expressed_length(parent2))
        if limit <= 0:
            return (Genotype(list(parent1.dna)), Genotype(list(parent2.dna)))
        rindex = random.randint(0, limit - 1)
        c1 = parent1.dna[:rindex] + parent2.dna[rindex:]
        c2 = parent2.dna[:rindex] + parent1.dna[rindex:]
        return (Genotype(c1), Genotype(c2))


class ExtensibleGrammaticalEvolutionRepresentation(GrammaticalEvolutionRepresentation):
    """GE variant that appends random codons when mapping exhausts the genome."""

    def __init__(self, grammar: Grammar, decider: SynthesisDecider, gene_length: int = 256):
        super().__init__(grammar, decider, gene_length=gene_length, extend_genotype=True)
