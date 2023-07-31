from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from geneticengine.core.decorators import get_gengy
from geneticengine.core.grammar import Grammar
from geneticengine.core.problems import Problem
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import CrossoverOperator
from geneticengine.core.representations.api import MutationOperator
from geneticengine.core.representations.api import Representation
from geneticengine.core.representations.tree.initializations import (
    InitializationMethodType,
)
from geneticengine.core.representations.tree.initializations import grow_method
from geneticengine.core.representations.tree.treebased import random_node
from geneticengine.core.tree import TreeNode
from geneticengine.core.utils import get_arguments, get_generic_parameters, is_terminal
from geneticengine.core.utils import get_generic_parameter
from geneticengine.core.utils import is_generic
from geneticengine.core.utils import is_generic_list
from geneticengine.core.utils import strip_annotations
from geneticengine.core.evaluators import Evaluator
from geneticengine.exceptions import GeneticEngineError
from geneticengine.metahandlers.base import is_metahandler

MAX_RAND_INT = 100000
MAX_VALUE = 10000000
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
    r: Source,
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


def random_individual_simple(
    r: Source,
    g: Grammar,
    starting_symbol: Any,
    current_genotype: Genotype | None = None,
    max_depth: int = 5,
) -> Genotype:  # In this method we let the random source use the left_overs to fill up the individual
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
            r.randint(0, MAX_RAND_INT) for _ in range(1000)
        ]  # Necessary to source from when a production rule runs out of genes.
        current_genotype = Genotype(dna)

    assert type(current_genotype) == Genotype
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

    # return random_individual_simple(r, g, starting_symbol, current_genotype, max_depth)
    return random_individual(r, g, starting_symbol, current_genotype, max_depth)


def mutate(r: Source, g: Grammar, ind: Genotype, max_depth: int, all_codons_equal_probability=False) -> Genotype:
    if all_codons_equal_probability:

        def weight(key):
            return len(ind.dna[key])

    else:

        def weight(key):
            return 1

    rkey = r.choice_weighted(
        list(key for key in ind.dna.keys() if (len(ind.dna[key]) > 0) and (key != LEFTOVER_KEY) and (key != "")),
        list(
            weight(key) for key in ind.dna.keys() if (len(ind.dna[key]) > 0) and (key != LEFTOVER_KEY) and (key != "")
        ),
    )
    dna = ind.dna
    clone = [i for i in dna[rkey]]
    rindex = r.randint(0, len(dna[rkey]) - 1)
    clone[rindex] = r.randint(0, MAX_RAND_INT)
    dna[rkey] = clone
    return Genotype(dna)


def per_codon_mutate(r: Source, g: Grammar, ind: Genotype, max_depth: int, codon_prob: float) -> Genotype:
    dna = ind.dna
    for key in dna.keys():
        if key != LEFTOVER_KEY and key != "":
            for i in range(len(dna[key])):
                if r.random_float(0, 1) < codon_prob:
                    dna[key][i] = r.randint(0, MAX_RAND_INT)
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
            c1[k] = p1.dna[k]
            c2[k] = p2.dna[k]
        else:
            c1[k] = p2.dna[k]
            c2[k] = p1.dna[k]
    return (Genotype(c1), Genotype(c2))


def phenotype_to_genotype(
    r: Source,
    g: Grammar,
    p: TreeNode,
    depth: int,
) -> Genotype:
    """An imperfect method that tries to reconstruct the genotype from the
    phenotype for the grow method.

    It is not possible to reconstruct integers and floats, due to the
    way integers and floats are constructed using a normalvariate
    function. However, the tree structure and node types are preserved.
    Therefore, this method can be used in the initialization process of
    trees.
    """
    assert p.gengy_distance_to_term <= depth
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

    non_terminals = g.non_terminals

    def filter_choices(possible_choices: list[type], depth):
        valid_productions = [vp for vp in possible_choices if g.get_distance_to_terminal(vp) <= depth]
        return valid_productions

    def find_choices_super(t: TreeNode, ttype, depth: int):
        supers = type(t).__mro__
        ttype_index = supers.index(ttype)
        i = 1
        while i <= ttype_index:
            randint = r.randint(1, MAX_VALUE)
            possible_choices = filter_choices(g.alternatives[ttype], depth)
            choice = possible_choices.index(supers[ttype_index - i])
            dna[str(ttype)].append(choice + randint * len(possible_choices))
            ttype = g.alternatives[ttype][choice]
            i += 1

    def randint_inverse(min: int, max: int, v: int) -> int:
        assert v >= min
        assert v <= max
        return v - min

    def random_float_inverse(min: float, max: float, v: float) -> int:
        k = round((max - min) / (v - min))
        return randint_inverse(1, MAX_VALUE, k)

    def normalvariate_inverse(mean: float, sigma: float, v: float):
        # z0 = (v - mean) / sigma
        # u1 = math.e ** ((z0**2) / (-2.0))
        # u2 = 0 if z0 > 0 else 0.5
        # return random_float_inverse(0.0, 1.0, u1), random_float_inverse(0.0, 1.0, u2)
        return r.randint(1, MAX_VALUE), r.randint(1, MAX_VALUE)

    def apply_inverse_metahandler(
        g: Grammar,
        depth: int,
        rec,
        base_type,
        instance,
        random_number: int,
    ) -> Any:
        """This method applies the inverse metahandler to use a custom
        generator for things of a given type.

        As an example, AnnotatedType[int, IntRange(3,10)] will use the
        IntRange.generate(r, recursive_generator). The generator is the
        annotation on the type ("__metadata__").
        """
        metahandler = base_type.__metadata__[0]
        return metahandler.inverse_generate(
            g,
            depth,
            rec,
            base_type,
            instance,
            random_number,
        )

    def reconstruct_genotype(t: Any, starting_symbol, depth: int, dna: dict[str, list[int]]):
        if type(t) not in [int, float, str, bool, list]:
            find_choices_super(t, starting_symbol, depth)
        if is_metahandler(starting_symbol):
            # reconstruct metahandler?
            x = apply_inverse_metahandler(g, depth, reconstruct_genotype, starting_symbol, t, r.randint(1, MAX_VALUE))
            # import IPython as ip
            # ip.embed()
            dna[str(strip_annotations(starting_symbol))].append(x)
        else:
            if is_terminal(type(t), non_terminals) and (not isinstance(t, list)):
                if isinstance(t, int) or isinstance(t, float):
                    x1, x2 = normalvariate_inverse(0, 100, t)
                    dna[str(type(t))].append(x1)
                    dna[str(type(t))].append(x2)
                if isinstance(t, bool):
                    if t:
                        dna[str(type(t))].append(0)
                    else:
                        dna[str(type(t))].append(1)
            else:
                if isinstance(t, list):
                    if depth > 1:
                        x = randint_inverse(1, depth, len(t))
                        dna[str(type(t))].append(x + depth * r.randint(1, MAX_VALUE))
                    children = [(type(obj), obj) for obj in t]
                else:
                    if not hasattr(t, "gengy_init_values"):
                        breakpoint()
                    children = [(typ[1], t.gengy_init_values[idx]) for idx, typ in enumerate(get_arguments(t))]
                for t, child in children:
                    dna = reconstruct_genotype(child, t, depth - 1, dna)
        return dna

    dna = reconstruct_genotype(p, g.starting_symbol, depth, dna)

    return Genotype(dna)


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
        k = self.randint(1, MAX_VALUE, prod)
        return 1 * (max - min) / k + min


def create_tree(
    g: Grammar,
    ind: Genotype,
    depth: int,
    initialization_mode: InitializationMethodType = grow_method,
) -> TreeNode:
    rand: Source = DynamicStructuredListWrapper(ind)
    return random_node(rand, g, depth, starting_symbol=g.starting_symbol, method=initialization_mode)


class DefaultDSGEMutation(MutationOperator[Genotype]):
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
        assert isinstance(representation, DynamicStructuredGrammaticalEvolutionRepresentation)
        return mutate(random_source, representation.grammar, genotype, representation.max_depth, False)


class EquiprobableCodonDSGEMutation(MutationOperator[Genotype]):
    """Chooses a random codon with equal probability.

    Finally, changes the value in that codon to another value.
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
        assert isinstance(representation, DynamicStructuredGrammaticalEvolutionRepresentation)
        return mutate(random_source, representation.grammar, genotype, representation.max_depth, True)


class PerCodonDSGEMutation(MutationOperator[Genotype]):
    """Chooses a random codon with a given probability.

    Then changes the value in that position.
    """

    def __init__(
        self,
        codon_probability: float = 0.2,
    ):
        self.codon_probability = codon_probability

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
        return per_codon_mutate(
            random_source,
            representation.grammar,
            genotype,
            representation.max_depth,
            self.codon_probability,
        )


class DefaultDSGECrossover(CrossoverOperator[Genotype]):
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
        return crossover(random_source, representation.grammar, g1, g2, representation.max_depth)


class DynamicStructuredGrammaticalEvolutionRepresentation(
    Representation[Genotype, TreeNode],
):
    """This version uses a list of lists of integers to represent individuals,
    based on non-terminal symbols.

    It delays computing the expansions that have enough depth to
    runtime.
    """

    def __init__(
        self,
        grammar: Grammar,
        max_depth: int,
        initialization_mode: InitializationMethodType = grow_method,
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
        return create_individual(r, self.grammar, max_depth=actual_depth)

    def genotype_to_phenotype(self, genotype: Genotype) -> TreeNode:
        return create_tree(
            self.grammar,
            genotype,
            self.max_depth,
            self.initialization_mode,
        )

    def phenotype_to_genotype(self, r: Source, phenotype: Any) -> Genotype:
        """Takes an existing program and adapts it to be used in the right
        representation."""
        return phenotype_to_genotype(r, self.grammar, phenotype, self.max_depth)

    def get_mutation(self) -> MutationOperator[Genotype]:
        return DefaultDSGEMutation()

    def get_crossover(self) -> CrossoverOperator[Genotype]:
        return DefaultDSGECrossover()
