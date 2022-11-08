from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import List
from typing import Protocol
from typing import Tuple
from typing import Type

from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import Representation
from geneticengine.core.representations.tree.initialization_methods import Initialization_Method, PI_Grow
from geneticengine.core.representations.tree.treebased import random_node
from geneticengine.core.tree import TreeNode


@dataclass
class Genotype:
    dna: list[int]


def random_individual(
    r: Source,
    g: Grammar,
    depth: int = 5,
    starting_symbol: Any = None,
    gene_size: int = 256,
) -> Genotype:
    return Genotype([r.randint(0, 10000) for _ in range(gene_size)])


def mutate(r: Source, g: Grammar, ind: Genotype, max_depth: int) -> Genotype:
    rindex = r.randint(0, len(ind.dna) - 1)
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
    rindex = r.randint(0, len(p1.dna) - 1)
    c1 = p1.dna[:rindex] + p2.dna[rindex:]
    c2 = p2.dna[:rindex] + p1.dna[rindex:]
    return (Genotype(c1), Genotype(c2))


class ListWrapper(Source):
    dna: list[int]
    index: int = 0
    
    def __init__(self, dna):
        self.dna = dna

    def randint(self, min: int, max: int, prod: str = "") -> int:
        self.index = (self.index + 1) % len(self.dna)
        v = self.dna[self.index]
        return v % (max - min + 1) + min

    def random_float(self, min: float, max: float, prod: str = "") -> float:
        k = self.randint(1, 100000000, prod)
        return 1 * (max - min) / k + min


def create_tree(g: Grammar, ind: Genotype, depth: int, method) -> TreeNode:
    rand: Source = ListWrapper(ind.dna)
    return random_node(rand, g, depth, g.starting_symbol, method=method)


class GrammaticalEvolutionRepresentation(Representation[Genotype]):
    """This representation uses a list of integers to guide the generation of trees in the phenotype."""

    def __init__(self, depth = None, gene_size=256, method: Initialization_Method = PI_Grow()) -> None:
        self.depth = depth
        self.gene_size = gene_size
        self.method = method

    def create_individual(self, r: Source, g: Grammar, depth: int) -> Genotype:
        self.depth = depth
        return random_individual(r, g, depth, gene_size=self.gene_size)

    def mutate_individual(
        self,
        r: Source,
        g: Grammar,
        ind: Genotype,
        depth: int,
        ty: type,
        specific_type: type | None = None,
        depth_aware_mut: bool = False,
    ) -> Genotype:
        return mutate(r, g, ind, depth)

    def crossover_individuals(
        self,
        r: Source,
        g: Grammar,
        i1: Genotype,
        i2: Genotype,
        depth: int,
        specific_type: type | None = None,
        depth_aware_co: bool = False,
    ) -> tuple[Genotype, Genotype]:
        return crossover(r, g, i1, i2, depth)

    def genotype_to_phenotype(self, g: Grammar, genotype: Genotype) -> TreeNode:
        return create_tree(g, genotype, self.depth, self.method.tree_init_method)


ge_representation = GrammaticalEvolutionRepresentation()
