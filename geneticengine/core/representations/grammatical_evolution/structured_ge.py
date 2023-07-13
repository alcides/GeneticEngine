from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from geneticengine.core.grammar import Grammar
from geneticengine.core.problems import Problem
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import CrossoverOperator
from geneticengine.core.representations.api import MutationOperator
from geneticengine.core.representations.api import Representation
from geneticengine.core.representations.tree.initializations import (
    InitializationMethodType,
)
from geneticengine.core.representations.tree.initializations import pi_grow_method
from geneticengine.core.representations.tree.treebased import random_node
from geneticengine.core.tree import TreeNode
from geneticengine.core.utils import get_arguments
from geneticengine.core.utils import is_generic
from geneticengine.core.utils import strip_annotations
from geneticengine.core.evaluators import Evaluator

GENE_SIZE = 100
MAX_RAND_INT = 10000000
INFRASTRUCTURE_KEY = "$infrastructure"


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

    # Implementer's decision: There are some random choices not related to productions.
    # E.g., the choice of production in PI-grow, metahandlers, etc.
    nodes.append(INFRASTRUCTURE_KEY)
    dna: dict[str, list[int]] = dict()
    for nodestr in nodes:
        dna[nodestr] = [r.randint(0, MAX_RAND_INT) for _ in range(GENE_SIZE)]

    return Genotype(dna)


def mutate(r: Source, g: Grammar, ind: Genotype, max_depth: int) -> Genotype:
    rkey = r.choice(list(ind.dna.keys()))
    rindex = r.randint(0, len(ind.dna[rkey]) - 1)

    dna = deepcopy(ind.dna)
    dna[rkey][rindex] = r.randint(0, MAX_RAND_INT)
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
            c1[k] = deepcopy(p1.dna[k])
            c2[k] = deepcopy(p2.dna[k])
        else:
            c1[k] = deepcopy(p2.dna[k])
            c2[k] = deepcopy(p1.dna[k])
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
        k = self.randint(1, MAX_RAND_INT, prod)
        return 1 * (max - min) / k + min


def create_tree(
    g: Grammar,
    ind: Genotype,
    depth: int,
    initialization_mode: InitializationMethodType = pi_grow_method,
) -> TreeNode:
    rand: Source = StructuredListWrapper(ind.dna)
    return random_node(rand, g, depth, g.starting_symbol, initialization_mode)


class DefaultSGEMutation(MutationOperator[Genotype]):
    """Chooses a random list, and a random position inside that list.

    Then changes the value in that position to another value.
    """

    def mutate(
        self,
        genotype: Genotype,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random_source: Source,
        index_in_population: int,
        generation: int,
    ) -> Genotype:
        assert isinstance(representation, StructuredGrammaticalEvolutionRepresentation)
        return mutate(
            random_source,
            representation.grammar,
            genotype,
            representation.max_depth,
        )


class DefaultSGECrossover(CrossoverOperator[Genotype]):
    """One-point crossover between the lists of lists."""

    def crossover(
        self,
        g1: Genotype,
        g2: Genotype,
        problem: Problem,
        representation: Representation,
        random_source: Source,
        index_in_population: int,
        generation: int,
    ) -> tuple[Genotype, Genotype]:
        assert isinstance(representation, StructuredGrammaticalEvolutionRepresentation)
        return crossover(random_source, representation.grammar, g1, g2, representation.max_depth)


class StructuredGrammaticalEvolutionRepresentation(Representation[Genotype, TreeNode]):
    """This version uses a list of lists of integers to represent individuals,
    based on non-terminal symbols."""

    def __init__(
        self,
        grammar: Grammar,
        max_depth: int,
        initialization_mode: InitializationMethodType = pi_grow_method,
    ):
        """
        Args:
            grammar (Grammar): The grammar to use in the mapping
            max_depth (int): the maximum depth when performing the mapping
            initialization_mode (InitializationMethodType): method to create individuals in the mapping (e.g., pi_grow,
                full, grow)
        """
        super().__init__(grammar, max_depth)

        self.initialization_mode = initialization_mode

    def create_individual(
        self,
        r: Source,
        depth: int | None = None,
        **kwargs,
    ) -> Genotype:
        actual_depth = depth or self.max_depth
        return random_individual(r, self.grammar, depth=actual_depth)

    def mutate_individual(
        self,
        r: Source,
        ind: Genotype,
        depth: int,
        ty: type,
        specific_type: type | None = None,
        depth_aware_mut: bool = False,
        **kwargs,
    ) -> Genotype:
        return mutate(r, self.grammar, ind, depth)

    def crossover_individuals(
        self,
        r: Source,
        i1: Genotype,
        i2: Genotype,
        depth: int,
        specific_type: type | None = None,
        depth_aware_co: bool = False,
        **kwargs,
    ) -> tuple[Genotype, Genotype]:
        return crossover(r, self.grammar, i1, i2, depth)

    def genotype_to_phenotype(self, genotype: Genotype) -> TreeNode:
        return create_tree(
            self.grammar,
            genotype,
            self.max_depth,
            self.initialization_mode,
        )

    def phenotype_to_genotype(self, phenotype: Any) -> Genotype:
        """Takes an existing program and adapts it to be used in the right
        representation."""
        raise NotImplementedError(
            "Reconstruction of genotype not supported in this representation.",
        )

    def get_mutation(self) -> MutationOperator[Genotype]:
        return DefaultSGEMutation()

    def get_crossover(self) -> CrossoverOperator[Genotype]:
        return DefaultSGECrossover()
