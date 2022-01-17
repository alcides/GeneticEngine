from dataclasses import dataclass
from typing import Any, List, Protocol, Tuple

from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import RandomSource, Source
from geneticengine.core.representations.api import Representation
from geneticengine.core.representations.tree.treebased import random_node
from geneticengine.core.tree import TreeNode


@dataclass
class Genotype:
    dna: List[int]
    depth: int


def random_individual(
    r: Source, g: Grammar, depth: int = 5, starting_symbol: Any = None
) -> Genotype:
    return Genotype([r.randint(0, 10000) for _ in range(256)], depth)


def mutate(r: Source, g: Grammar, ind: Genotype, max_depth: int) -> Genotype:
    rindex = r.randint(0, 255)
    clone = [i for i in ind.dna]
    clone[rindex] = r.randint(0, 10000)
    return Genotype(clone, ind.depth)


def crossover(
    r: Source, g: Grammar, p1: Genotype, p2: Genotype, max_depth: int
) -> Tuple[Genotype, Genotype]:
    rindex = r.randint(0, 255)
    c1 = p1.dna[:rindex] + p2.dna[rindex:]
    c2 = p2.dna[:rindex] + p1.dna[rindex:]
    return (Genotype(c1, p1.depth), Genotype(c2, p2.depth))


@dataclass
class ListWrapper(Source):
    list: List[int]
    index: int = 0

    def randint(self, min, max) -> int:
        self.index = self.index + 1 % len(self.list)
        v = self.list[self.index]
        return v % (max - min + 1) + min

    def random_float(self, min, max) -> float:
        k = self.randint(1, 100000000)
        return 1 * (max - min) / k + min


def create_tree(g: Grammar, ind: Genotype) -> TreeNode:
    rand: Source = ListWrapper(ind.dna)
    return random_node(rand, g, ind.depth, g.starting_symbol)


class GrammaticalEvolutionRepresentation(Representation[Genotype]):
    def create_individual(self, r: Source, g: Grammar, depth: int) -> Genotype:
        return random_individual(r, g, depth)

    def mutate_individual(
        self, r: Source, g: Grammar, ind: Genotype, depth: int
    ) -> Genotype:
        return mutate(r, g, ind, depth)

    def crossover_individuals(
        self, r: Source, g: Grammar, i1: Genotype, i2: Genotype, int
    ) -> Tuple[Genotype, Genotype]:
        return crossover(r, g, i1, i2, int)

    def genotype_to_phenotype(self, g: Grammar, genotype: Genotype) -> TreeNode:
        return create_tree(g, genotype)


ge_representation = GrammaticalEvolutionRepresentation()
