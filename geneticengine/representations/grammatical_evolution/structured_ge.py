from __future__ import annotations

from copy import deepcopy
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
from geneticengine.grammar.utils import get_arguments
from geneticengine.grammar.utils import is_generic
from geneticengine.grammar.utils import strip_annotations

INFRASTRUCTURE_KEY = "$infrastructure"


@dataclass
class Genotype:
    dna: dict[str, list[int]]


class StructuredListWrapper(RandomSource):
    dna: dict[str, list[int]]
    indexes: dict[str, int]

    def __init__(self, dna):
        self.dna = dna
        indexes = dict()
        for k in dna.keys():
            indexes[k] = 0
        self.indexes = indexes

    def randint(self, min: int, max: int, prod: str = INFRASTRUCTURE_KEY) -> int:
        self.indexes[prod] = (self.indexes[prod] + 1) % len(self.dna[prod])
        v = self.dna[prod][self.indexes[prod]]
        return v % (max - min + 1) + min

    def random_float(
        self,
        min: float,
        max: float,
        prod: str = INFRASTRUCTURE_KEY,
    ) -> float:
        k = self.randint(1, sys.maxsize)
        return 1 * (max - min) / k + min


class StructuredGrammaticalEvolutionRepresentation(
    Representation[Genotype, TreeNode],
    RepresentationWithMutation[Genotype],
    RepresentationWithCrossover[Genotype],
):
    """This version uses a list of lists of integers to represent individuals,
    based on non-terminal symbols."""

    def __init__(
        self,
        grammar: Grammar,
        decider: SynthesisDecider,
        gene_length: int = 256,
    ):
        """
        Args:
            grammar (Grammar): The grammar to use in the mapping
            max_depth (int): the maximum depth when performing the mapping
        """
        self.grammar = grammar
        self.decider = decider
        self.gene_length = gene_length

    def create_genotype(self, random: RandomSource, **kwargs) -> Genotype:
        nodes = [str(node) for node in self.grammar.all_nodes]
        for node in self.grammar.all_nodes:
            arguments = get_arguments(node)
            for _, arg in arguments:
                if is_generic(arg):
                    nodes.append(str(arg))
                base_type = str(strip_annotations(arg))
                if base_type not in nodes:
                    nodes.append(base_type)

        # Implementer's decision: There are some random choices not related to productions.
        # E.g., the choice of production in PI-grow, metahandlers, etc.
        nodes.append(INFRASTRUCTURE_KEY)
        dna: dict[str, list[int]] = dict()
        for nodestr in nodes:
            dna[nodestr] = [random.randint(0, sys.maxsize) for _ in range(self.gene_length)]

        return Genotype(dna)

    def genotype_to_phenotype(self, genotype: Genotype) -> TreeNode:
        rand: RandomSource = StructuredListWrapper(genotype.dna)
        return random_node(rand, self.grammar, self.grammar.starting_symbol, self.decider)

    def mutate(self, random: RandomSource, genotype: Genotype, **kwargs) -> Genotype:
        rkey = random.choice(list(genotype.dna.keys()))
        rindex = random.randint(0, len(genotype.dna[rkey]) - 1)

        dna = deepcopy(genotype.dna)
        dna[rkey][rindex] = random.randint(0, sys.maxsize)
        return Genotype(dna)

    def crossover(
        self,
        random: RandomSource,
        parent1: Genotype,
        parent2: Genotype,
        **kwargs,
    ) -> tuple[Genotype, Genotype]:
        keys = parent1.dna.keys()

        mask = [(k, random.random_bool()) for k in keys]
        c1 = dict()
        c2 = dict()
        for k, b in mask:
            if b:
                c1[k] = deepcopy(parent1.dna[k])
                c2[k] = deepcopy(parent2.dna[k])
            else:
                c1[k] = deepcopy(parent2.dna[k])
                c2[k] = deepcopy(parent1.dna[k])
        return (Genotype(c1), Genotype(c2))
