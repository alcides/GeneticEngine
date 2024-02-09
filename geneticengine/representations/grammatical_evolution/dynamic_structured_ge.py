from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import sys
from typing import Any
from geneticengine.exceptions import GeneticEngineError
from geneticengine.grammar.decorators import get_gengy

from geneticengine.grammar.grammar import Grammar
from geneticengine.grammar.metahandlers.base import is_metahandler
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import (
    RepresentationWithCrossover,
    RepresentationWithMutation,
    SolutionRepresentation,
)
from geneticengine.representations.tree.initializations import (
    InitializationMethodType,
)
from geneticengine.representations.tree.initializations import pi_grow_method
from geneticengine.representations.tree.treebased import random_individual, random_node
from geneticengine.solutions.tree import TreeNode
from geneticengine.grammar.utils import get_arguments
from geneticengine.grammar.utils import is_generic
from geneticengine.grammar.utils import strip_annotations
from geneticengine.grammar.utils import get_generic_parameters
from geneticengine.grammar.utils import get_generic_parameter
from geneticengine.grammar.utils import is_generic_list

INFRASTRUCTURE_KEY = "$infrastructure"
LEFTOVER_KEY = "$leftovers"
MAX_RAND_LIST_SIZE = 10


@dataclass
class Genotype:
    dna: dict[str, list[int]]

    def register_production(self, prod_index: int, starting_symbol):
        if str(starting_symbol) in self.dna.keys():
            self.dna[str(starting_symbol)].append(prod_index)
        else:
            self.dna[str(starting_symbol)] = [prod_index]


def filter_choices(possible_choices: list[type], g: Grammar, depth, starting_symbol):
    valid_productions = [vp for vp in possible_choices if g.get_distance_to_terminal(vp) <= depth]
    if not valid_productions:
        raise GeneticEngineError(
            "No productions for non-terminal node with type: {} in depth {} (minimum required: {}).".format(
                starting_symbol,
                depth,
                str(
                    [(vp, g.distanceToTerminal[vp]) for vp in possible_choices],
                ),
            ),
        )
    return valid_productions


def assert_depth_error(max_depth, g, starting_symbol):
    if max_depth < g.get_distance_to_terminal(starting_symbol):
        raise GeneticEngineError(
            "There will be no depth sufficient for symbol {} in this grammar (provided: {}, required: {}).".format(
                starting_symbol,
                max_depth,
                g.get_distance_to_terminal(
                    starting_symbol,
                ),
            ),
        )


def random_individual(
    r: RandomSource,
    g: Grammar,
    starting_symbol: Any,
    current_genotype: Genotype | None = None,
    max_depth: int = 5,
) -> Genotype:

    # This whole method seems cumbersome. Why not just create an empty genotype and let it be sourced through when
    # mapping from genotype to phenotype?
    if current_genotype is None:
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
            dna[nodestr] = list()
        dna[LEFTOVER_KEY] = [
            r.randint(0, sys.maxsize) for _ in range(1000)
        ]  # Necessary to source from when a production rule runs out of genes.
        current_genotype = Genotype(dna)
    assert type(current_genotype) == Genotype

    if starting_symbol in [int, float, str, bool]:
        val = r.randint(0, sys.maxsize)
        current_genotype.register_production(val, starting_symbol)
    elif starting_symbol in g.alternatives:
        assert_depth_error(max_depth, g, starting_symbol)
        productions = g.alternatives[starting_symbol]
        valid_productions = filter_choices(
            productions,
            g,
            max_depth - int(g.expansion_depthing),
            starting_symbol,
        )
        if any(["weight" in get_gengy(p) for p in valid_productions]):
            weights = [get_gengy(p).get("weight", 1.0) for p in valid_productions]
            prod = r.choice_weighted(
                valid_productions,
                weights,
                str(starting_symbol),
            )
        else:
            prod = r.choice(valid_productions)
        prod_index = valid_productions.index(prod)
        current_genotype.register_production(prod_index, starting_symbol)
        current_genotype = random_individual(
            r,
            g,
            prod,
            current_genotype,
            max_depth - int(g.expansion_depthing),
        )
    elif is_generic_list(starting_symbol):
        new_type = get_generic_parameter(starting_symbol)
        list_length = r.randint(1, MAX_RAND_LIST_SIZE)
        current_genotype.register_production(list_length, int)
        for _ in range(list_length):
            current_genotype = random_individual(
                r,
                g,
                new_type,
                current_genotype,
                max_depth,
            )
    elif is_metahandler(
        starting_symbol,
    ):
        # No need to go down one in depth as this is the argument of a treenode and therefore already has an adjusted
        # depth.
        if is_generic_list(get_generic_parameter(starting_symbol)):
            new_type = get_generic_parameter(starting_symbol)
            current_genotype = random_individual(
                r,
                g,
                new_type,
                current_genotype,
                max_depth,
            )
        else:
            new_type = strip_annotations(starting_symbol)
            current_genotype = random_individual(
                r,
                g,
                new_type,
                current_genotype,
                max_depth,
            )
    elif is_generic(starting_symbol):

        def recgen(v):
            return random_individual(
                r,
                g,
                v,
                current_genotype,
                max_depth,
            )

        g_args = get_generic_parameters(starting_symbol)
        assert tuple(recgen(a) for a in g_args)
    else:
        assert_depth_error(max_depth, g, starting_symbol)
        if starting_symbol not in g.all_nodes:
            raise GeneticEngineError(
                f"Symbol {starting_symbol} not in grammar rules.",
            )

        args = get_arguments(starting_symbol)
        for _, argt in args:
            current_genotype = random_individual(
                r,
                g,
                argt,
                current_genotype,
                max_depth - 1,
            )

    return current_genotype


class DynamicStructuredListWrapper(RandomSource):
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
        ):
            # We don't have a wrapper function, but we add elements to each list when there are no genes left.
            # These are sourced from the "left_overs" in the dna.
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
        k = self.randint(1, sys.maxsize, prod)
        return 1 * (max - min) / k + min


def create_tree(
    g: Grammar,
    ind: Genotype,
    depth: int,
    initialization_mode: InitializationMethodType = pi_grow_method,
) -> TreeNode:
    rand: RandomSource = DynamicStructuredListWrapper(ind)
    return random_node(rand, g, depth, g.starting_symbol, initialization_mode)


class DynamicStructuredGrammaticalEvolutionRepresentation(
    SolutionRepresentation[Genotype, TreeNode],
    RepresentationWithMutation[Genotype],
    RepresentationWithCrossover[Genotype],
):
    """This version uses a list of lists of integers to represent individuals,
    based on non-terminal symbols."""

    def __init__(
        self,
        grammar: Grammar,
        max_depth: int,  # TODO: parameterize
        gene_length: int = 256,
        initialization_mode: InitializationMethodType = pi_grow_method,
    ):
        """
        Args:
            grammar (Grammar): The grammar to use in the mapping
            max_depth (int): the maximum depth when performing the mapping
            initialization_mode (InitializationMethodType): method to create individuals in the mapping
                (e.g., pi_grow, full, grow)
        """
        self.grammar = grammar
        self.max_depth = max_depth
        self.gene_length = gene_length
        self.initialization_mode = initialization_mode

    def instantiate(self, random: RandomSource, **kwargs) -> Genotype:
        return random_individual(random, self.grammar, self.grammar.starting_symbol, None, self.max_depth)

    def map(self, genotype: Genotype) -> TreeNode:
        rand: RandomSource = DynamicStructuredListWrapper(genotype)
        return random_node(rand, self.grammar, self.max_depth, self.grammar.starting_symbol, self.initialization_mode)

    def mutate(self, random: RandomSource, internal: Genotype, **kwargs) -> Genotype:
        dna = deepcopy(internal.dna)
        rkey = random.choice(list(internal.dna.keys()))
        if internal.dna[rkey]:
            rindex = random.randint(0, len(internal.dna[rkey]) - 1)
            dna[rkey][rindex] = random.randint(0, sys.maxsize)
        return Genotype(dna)

    def crossover(
        self, random: RandomSource, parent1: Genotype, parent2: Genotype, **kwargs,
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
