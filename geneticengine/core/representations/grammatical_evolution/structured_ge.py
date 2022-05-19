from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
from typing import List
from typing import Protocol
from typing import Tuple
from typing import Type

from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import Representation
from geneticengine.core.representations.tree.treebased import Grow
from geneticengine.core.representations.tree.treebased import random_node
from geneticengine.core.tree import TreeNode
from geneticengine.core.utils import get_arguments
from geneticengine.core.utils import is_generic
from geneticengine.core.utils import strip_annotations

LIST_SIZE = 100


@dataclass
class Genotype:
    dna: dict[str, list[int]]


def random_individual(
    r: Source,
    g: Grammar,
    depth: int = 5,
    starting_symbol: Any = None,
) -> Genotype:
    nodes = [str(node) for node in g.all_nodes]
    for node in g.all_nodes:
        arguments = get_arguments(node)
        for _, arg in arguments:
            if is_generic(arg):
                nodes.append(str(arg))
            base_type = str(strip_annotations(arg))
            if base_type not in nodes:
                nodes.append(base_type)

    dna: Dict[str, List[int]] = dict()
    for nodestr in nodes:
        dna[nodestr] = [r.randint(0, 10000) for _ in range(LIST_SIZE)]

    return Genotype(dna)


def mutate(r: Source, g: Grammar, ind: Genotype, max_depth: int) -> Genotype:
    rkey = r.choice(list(ind.dna.keys()))
    dna = ind.dna
    clone = [i for i in dna[rkey]]
    rindex = r.randint(0, len(dna[rkey])) - 1
    clone[rindex] = r.randint(0, 10000)
    dna[rkey] = clone
    return Genotype(dna)


def crossover(
    r: Source,
    g: Grammar,
    p1: Genotype,
    p2: Genotype,
    max_depth: int,
) -> tuple[Genotype, Genotype]:
    keys = p1.dna.keys()
    mask = [(k, r.random_bool()) for k in keys]
    c1 = dict()
    c2 = dict()
    for k, b in mask:
        if b:
            c1[k] = p1.dna[k]
            c2[k] = p2.dna[k]
        else:
            c1[k] = p2.dna[k]
            c2[k] = p1.dna[k]
    return (Genotype(c1), Genotype(c2))


class StructuredListWrapper(Source):
    dna: dict[str, list[int]]
    indexes: dict[str, int]

    def __init__(self, dna):
        self.dna = dna
        indexes = dict()
        for k in dna.keys():
            indexes[k] = 0
        self.indexes = indexes

    def randint(self, min: int, max: int, prod: str = "") -> int:
        self.indexes[prod] = (self.indexes[prod] + 1) % len(self.dna[prod])
        v = self.dna[prod][self.indexes[prod]]
        return v % (max - min + 1) + min

    def random_float(self, min: float, max: float, prod: str = "") -> float:
        k = self.randint(1, 100000000, prod)
        return 1 * (max - min) / k + min


def create_tree(g: Grammar, ind: Genotype, depth: int) -> TreeNode:
    rand: Source = StructuredListWrapper(ind.dna)
    return random_node(rand, g, depth, g.starting_symbol, method=Grow)


class StructuredGrammaticalEvolutionRepresentation(Representation[Genotype]):
    depth: int

    def create_individual(self, r: Source, g: Grammar, depth: int) -> Genotype:
        self.depth = depth
        return random_individual(r, g, depth)

    def mutate_individual(
        self,
        r: Source,
        g: Grammar,
        ind: Genotype,
        depth: int,
        ty: type,
    ) -> Genotype:
        return mutate(r, g, ind, depth)

    def crossover_individuals(
        self,
        r: Source,
        g: Grammar,
        i1: Genotype,
        i2: Genotype,
        depth: int,
    ) -> tuple[Genotype, Genotype]:
        return crossover(r, g, i1, i2, depth)

    def genotype_to_phenotype(self, g: Grammar, genotype: Genotype) -> TreeNode:
        return create_tree(g, genotype, self.depth)


sge_representation = StructuredGrammaticalEvolutionRepresentation()
