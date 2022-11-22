from __future__ import annotations
from copy import deepcopy

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

MAX_RAND_INT = 100000

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
    return Genotype([r.randint(0, MAX_RAND_INT) for _ in range(gene_size)])


def standard_mutate(r: Source, g: Grammar, ind: Genotype, max_depth: int, mutation_method) -> Genotype:
    dna = ind.dna
    rindex = r.randint(0, len(dna) - 1)
    dna[rindex] = r.randint(0, MAX_RAND_INT)
    return Genotype(dna)

def per_codon_mutate(r: Source, g: Grammar, ind: Genotype, max_depth: int, codon_prob) -> Genotype:
    dna = ind.dna
    for i in range(len(dna)):
        if r.random_float(0,1) < codon_prob:
            dna[i] = r.randint(0,MAX_RAND_INT)
    return Genotype(dna)
    
def mutate(r: Source, g: Grammar, ind: Genotype, max_depth: int, mutation_method, codon_prob) -> Genotype:
    if mutation_method == 'per_codon_mutate':
        return per_codon_mutate(r, g, ind, max_depth, codon_prob)
    else:
        return standard_mutate(r, g, ind, max_depth, mutation_method)
        


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
    """This representation uses a list of integers to guide the generation of trees in the phenotype.
        
    You can specify the [mutation_method] as follows:
    - One mutation with all codons equal probabilities: all_codons_equal_prob
    - Mutation possibility for each codon (this also allows you to specify the [codon_prob]): per_codon_mutate
    """

    def __init__(self, depth = None, gene_size=256, method: Initialization_Method = PI_Grow(), mutation_method = 'all_codons_equal_prob', codon_prob = 0.05) -> None:
        self.depth = depth
        self.gene_size = gene_size
        self.method = method
        self.mutation_method = mutation_method
        self.codon_prob = codon_prob

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
        new_ind = mutate(r, g, deepcopy(ind), depth, self.mutation_method, self.codon_prob)
        return new_ind

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
        return crossover(r, g, deepcopy(i1), deepcopy(i2), depth)

    def genotype_to_phenotype(self, g: Grammar, genotype: Genotype) -> TreeNode:
        return create_tree(g, genotype, self.depth, self.method.tree_init_method)


ge_representation = GrammaticalEvolutionRepresentation()
