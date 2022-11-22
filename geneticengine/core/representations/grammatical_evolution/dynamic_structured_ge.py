from __future__ import annotations

from copy import deepcopy
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
MAX_RAND_LIST_SIZE = 10


LEFTOVER_KEY = "$leftovers"


@dataclass
class Genotype:
    dna: dict[str, list[int]]

    def register_production(self, prod_index: int, starting_symbol):
        if str(starting_symbol) in self.dna.keys():
            self.dna[str(starting_symbol)].append(prod_index)
        else:
            self.dna[str(starting_symbol)] = [prod_index]
    
    def pretty_print(self) -> str:
        s = ''
        for key in self.dna.keys():
            if key != LEFTOVER_KEY:
                s += f"{key}: {self.dna[key]} \n"
        return s

def filter_choices(possible_choices: list[type], g: Grammar, depth, starting_symbol):
    valid_productions = [
        vp for vp in possible_choices if g.get_distance_to_terminal(vp) <= depth
    ]
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
    r: Source,
    g: Grammar,
    starting_symbol: Any,
    current_genotype: Genotype | None = None,
    max_depth: int = 5,
) -> Genotype: 
    if current_genotype == None:
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
            r.randint(0, MAX_RAND_INT) for _ in range(1000)
        ]  # Necessary to source from when a production rule runs out of genes.
        current_genotype = Genotype(dna)
    assert type(current_genotype) == Genotype

    if starting_symbol in [int, float, str, bool]:
        val = r.randint(0, MAX_RAND_INT)
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
    ):  # No need to go down one in depth as this is the argument of a treenode and therefore already has an adjusted depth.
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



def create_individual(
    r: Source,
    g: Grammar,
    starting_symbol: Any = None,
    current_genotype: Genotype | None = None,
    max_depth: int = 5,
) -> Genotype:
    if not starting_symbol:
        starting_symbol = g.starting_symbol

    return random_individual(r, g, starting_symbol, current_genotype, max_depth)


def standard_mutate(r: Source, g: Grammar, ind: Genotype, max_depth: int, mutation_method) -> Genotype:
    if mutation_method == 'all_codons_equal_prob':
        weight = lambda key: len(ind.dna[key])
    else:
        weight = lambda key: 1
    
    rkey = r.choice_weighted(
        list(
            key
            for key in ind.dna.keys()
            if (len(ind.dna[key]) > 0) and (key != LEFTOVER_KEY) and (key != '')
        ),
        list(weight(key) for key in ind.dna.keys() if (len(ind.dna[key]) > 0) and (key != LEFTOVER_KEY) and (key != ''))
    )
    dna = ind.dna
    clone = [i for i in dna[rkey]]
    rindex = r.randint(0, len(dna[rkey]) - 1)
    clone[rindex] = r.randint(0, MAX_RAND_INT)
    dna[rkey] = clone
    return Genotype(dna)

def per_codon_mutate(r: Source, g: Grammar, ind: Genotype, max_depth: int, codon_prob) -> Genotype:
    dna = ind.dna
    for key in dna.keys():
        if key != LEFTOVER_KEY and key != '':
            for i in range(len(dna[key])):
                if r.random_float(0,1) < codon_prob:
                    dna[key][i] = r.randint(0,MAX_RAND_INT)
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
    keys = p1.dna.keys()  # Leave leftovers in as it doesn't matter
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


class DynamicStructuredListWrapper(Source):
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
            self.indexes[prod] = 0

    def randint(self, min: int, max: int, prod: str = "") -> int:
        self.register_index(prod)
        if self.indexes[prod] >= len(
            self.ind.dna[prod],
        ):  # We don't have a wrapper function, but we add elements to each list when there are no genes left. These are sourced from the "left_overs" (LEFTOVER_KEY) in the dna.
            self.ind.register_production(
                self.ind.dna[LEFTOVER_KEY][self.indexes[LEFTOVER_KEY]],
                prod,
            )
            self.ind.dna[LEFTOVER_KEY].append(self.ind.dna[LEFTOVER_KEY].pop(0))
        v = self.ind.dna[prod][self.indexes[prod] - 1]
        return v % (max - min + 1) + min

    def random_float(self, min: float, max: float, prod: str = "") -> float:
        k = self.randint(1, 100000000, prod)
        return 1 * (max - min) / k + min


def create_tree(g: Grammar, ind: Genotype, depth: int, method) -> TreeNode:
    rand: Source = DynamicStructuredListWrapper(ind)
    return random_node(rand, g, depth, g.starting_symbol, method=method)


class DynamicStructuredGrammaticalEvolutionRepresentation(Representation[Genotype]):
    """This version uses a list of lists of integers to represent individuals, based on non-terminal
    symbols. It delays computing the expansions that have enough depth to runtime.

    You can specify the [mutation_method] as follows:
    - One mutation with all codons equal probabilities: all_codons_equal_prob
    - One mutation with all genes equal probabilities: all_genes_equal_prob
    - Mutation possibility for each codon (this also allows you to specify the [codon_prob]): per_codon_mutate
    """

    def __init__(self, depth = None, method: Initialization_Method = PI_Grow(), mutation_method = 'all_codons_equal_prob', codon_prob = 0.05) -> None:
        self.depth = depth
        self.method = method
        self.mutation_method = mutation_method
        self.codon_prob = codon_prob
    
    def create_individual(self, r: Source, g: Grammar, depth: int) -> Genotype:
        self.depth = depth
        return create_individual(r, g, max_depth=depth)

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


dsge_representation = DynamicStructuredGrammaticalEvolutionRepresentation()
