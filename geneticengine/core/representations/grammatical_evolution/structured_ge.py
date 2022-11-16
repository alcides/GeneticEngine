from __future__ import annotations
import copy

from dataclasses import dataclass
from typing import Any

from geneticengine.core.decorators import get_gengy
from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import Representation
from geneticengine.core.representations.tree.initialization_methods import Initialization_Method
from geneticengine.core.representations.tree.treebased import PI_Grow
from geneticengine.core.representations.tree.treebased import random_node
from geneticengine.core.tree import TreeNode
from geneticengine.core.utils import get_arguments
from geneticengine.core.utils import get_generic_parameter
from geneticengine.core.utils import is_generic
from geneticengine.core.utils import is_generic_list
from geneticengine.core.utils import strip_annotations
from geneticengine.exceptions import GeneticEngineError
from geneticengine.metahandlers.base import is_metahandler

MAX_RAND_INT = 100000

LEFTOVER_KEY = "$leftovers"


@dataclass
class Genotype:
    dna: dict[str, list[int]]

    def register_production(self, prod_index: int, starting_symbol):
        if str(starting_symbol) in self.dna.keys():
            self.dna[str(starting_symbol)].append(prod_index)
        else:
            self.dna[str(starting_symbol)] = [prod_index]

def random_individual(
    r: Source,
    g: Grammar,
    depth: int = 5,
    starting_symbol: Any = None,
    gene_size: int = 256
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

    dna: dict[str, list[int]] = dict()
    for nodestr in nodes:
        dna[nodestr] = [r.randint(0, MAX_RAND_INT) for _ in range(gene_size)]
    dna[LEFTOVER_KEY] = [
        r.randint(0, MAX_RAND_INT) for _ in range(gene_size)
    ]  # Necessary to source from when a production rule runs out of genes.
    current_genotype = Genotype(dna)

    assert type(current_genotype) == Genotype
    return current_genotype


def mutate(r: Source, g: Grammar, ind: Genotype, max_depth: int) -> Genotype:
    rkey = r.choice(
        list(
            key
            for key in ind.dna.keys()
            if (key != LEFTOVER_KEY)
        ),
    )
    dna = copy.deepcopy(ind.dna)
    clone = [i for i in dna[rkey]]
    rindex = r.randint(0, len(dna[rkey]) - 1)
    clone[rindex] = r.randint(0, MAX_RAND_INT)
    dna[rkey] = clone
    return Genotype(dna)


def crossover(
    r: Source,
    g: Grammar,
    p1: Genotype,
    p2: Genotype,
    max_depth: int,
) -> tuple[Genotype, Genotype]:
    keys = p1.dna.keys()  # Leave leftovers in as it doesn't matter
    mask = [(k, r.random_bool()) for k in keys]
    c1 = dict()
    c2 = dict()
    for k, b in mask:
        if b:
            c1[k] = copy.deepcopy(p1.dna[k])
            c2[k] = copy.deepcopy(p2.dna[k])
        else:
            c1[k] = copy.deepcopy(p2.dna[k])
            c2[k] = copy.deepcopy(p1.dna[k])
    return (Genotype(c1), Genotype(c2))


class StructuredListWrapper(Source):
    ind: Genotype
    indexes: dict[str, int]

    def __init__(self, ind: Genotype):
        self.ind = ind
        indexes = dict()
        for k in ind.dna.keys():
            indexes[k] = 0
        self.indexes = indexes

    def register_index(self, prod):
        if prod in self.indexes.keys():
            self.indexes[prod] += 1
        else:
            self.ind.dna[prod] = list()
            self.indexes[prod] = 1

    def randint(self, min: int, max: int, prod: str = "") -> int:
        self.register_index(prod)
        if self.indexes[prod] >= len(
            self.ind.dna[prod],
        ):  # We don't have a wrapper function, but we add elements to each list when there are no genes left. These are sourced from the "left_overs" in the dna.
            self.indexes[LEFTOVER_KEY] = (self.indexes[LEFTOVER_KEY] + 1) % len(
                self.ind.dna[LEFTOVER_KEY],
            )
            self.ind.register_production(
                self.ind.dna[LEFTOVER_KEY][self.indexes[LEFTOVER_KEY]],
                prod,
            )
        v = self.ind.dna[prod][self.indexes[prod] - 1]
        return v % (max - min + 1) + min

    def random_float(self, min: float, max: float, prod: str = "") -> float:
        k = self.randint(1, 25600000, prod)
        return 1 * (max - min) / k + min


def create_tree(g: Grammar, ind: Genotype, depth: int, method) -> TreeNode:
    rand: Source = StructuredListWrapper(ind)
    return random_node(rand, g, depth, g.starting_symbol, method=method)


class StructuredGrammaticalEvolutionRepresentation(Representation[Genotype]):
    """This version uses a list of lists of integers to represent individuals, based on non-terminal
    symbols."""

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


sge_representation = StructuredGrammaticalEvolutionRepresentation()
