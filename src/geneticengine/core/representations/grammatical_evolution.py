from dataclasses import dataclass
from typing import Any, List, Protocol, Tuple

from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import RandomSource, Source
from geneticengine.core.representations.base import Representation
from geneticengine.core.representations.treebased import random_node
from geneticengine.core.tree import TreeNode


@dataclass
class Genotype:
    dna: List[int]
    depth: int


def random_individual(
    r: RandomSource, g: Grammar, depth: int = 5, starting_symbol: Any = None
) -> Genotype:
    return Genotype([r.randint(0, 10000) for _ in range(256)], depth)


def mutate(r: RandomSource, g: Grammar, ind: Genotype, max_depth: int) -> Genotype:
    rindex = r.randint(0, 255)
    clone = [i for i in ind.dna]
    clone[rindex] = r.randint(0, 10000)
    return Genotype(clone, ind.depth)


def crossover(
    r: RandomSource, g: Grammar, p1: Genotype, p2: Genotype, max_depth: int
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
        v = self.list[self.index]
        self.index = self.index + 1 % len(self.list)
        return v % (max - min + 1) + min

    def random_float(self, min, max) -> float:
        k = self.randint(1, 100000000)
        return 1 * (max - min) / k + min


def create_tree(g: Grammar, ind: Genotype) -> TreeNode:
    rand: Source = ListWrapper(ind.dna)
    return random_node(rand, g, ind.depth, g.starting_symbol)


ge_representation = Representation(
    create_individual=random_individual,
    mutate_individual=mutate,
    crossover_individuals=crossover,
    genotype_to_phenotype=create_tree,
)
