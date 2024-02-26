from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
from functools import reduce
import sys

from typing import Any
from typing import TypeVar

from geneticengine.grammar.grammar import Grammar
from geneticengine.grammar.metahandlers.base import is_metahandler
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import (
    RepresentationWithCrossover,
    RepresentationWithMutation,
    Representation,
)
from geneticengine.representations.tree.initializations import (
    InitializationMethodType,
    grow_method,
    pi_grow_method,
)
from geneticengine.representations.tree.initializations import mk_save_init
from geneticengine.representations.tree.utils import relabel_nodes
from geneticengine.representations.tree.utils import relabel_nodes_of_trees
from geneticengine.solutions.tree import GengyList, TreeNode
from geneticengine.grammar.utils import (
    get_arguments,
    get_generic_parameters,
    is_builtin_class_instance,
    is_generic_list,
    is_generic_tuple,
)
from geneticengine.grammar.utils import get_generic_parameter
from geneticengine.grammar.utils import has_annotated_crossover
from geneticengine.grammar.utils import has_annotated_mutation
from geneticengine.grammar.utils import is_abstract
from geneticengine.exceptions import GeneticEngineError

T = TypeVar("T")


@dataclass
class SynthesisContext:
    depth: int
    nodes: int
    expansions: int


class SynthesisDecider(ABC):
    def random_int(self) -> int: ...
    def random_float(self) -> float: ...
    def random_str(self) -> str: ...
    def random_bool(self) -> bool: ...
    def random_tuple(self, types) -> tuple: ...
    def random_list(self, type) -> list[Any]: ...
    def choose_alternatives(self, alternatives: list[T]) -> T: ...


class BasicSynthesisDecider(SynthesisDecider):
    def __init__(self, random: RandomSource, grammar: Grammar):
        self.random = random
        self.grammar = grammar

    def random_int(self) -> int:
        max_int = sys.maxsize
        min_int = -sys.maxsize
        val = self.random.normalvariate(0, max_int / 100)
        val = round(val)
        return max(min(val, max_int), min_int)

    def random_float(self) -> float:
        max_float = sys.float_info.max
        min_float = -sys.float_info.max
        valf = self.random.normalvariate(0, 1)
        return max(min(valf, max_float), min_float)

    def random_str(self) -> str:
        length = int(abs(round(self.random.normalvariate(0, 10), 0)))
        return str(chr(self.random.randint(32, 128)) for _ in range(length))

    def random_bool(self) -> bool:
        return self.random.random_bool()

    def random_tuple(self, types) -> tuple:
        els = (random_tree(random=self.random, grammar=self.grammar, starting_symbol=t, decider=self) for t in types)
        return reduce(lambda x, y: x + y, els)

    def random_list(self, type) -> list[Any]:
        length = int(abs(round(self.random.normalvariate(0, 10), 0)))
        return [
            random_tree(random=self.random, grammar=self.grammar, starting_symbol=type, decider=self)
            for _ in range(length)
        ]

    def choose_alternatives(self, alternatives: list[T]) -> T:
        assert len(alternatives) > 0, "No alternatives presented"
        return alternatives[0]  # TODO


def relabel(f):
    """Decorator that relabels nodes when they are generated."""

    def g(*args, **kwargs):
        n = f(*args, **kwargs)
        grammar = kwargs["grammar"]
        starting_symbol = kwargs["starting_symbol"]

        if isinstance(n, list):
            n = GengyList(starting_symbol, n)
        relabel_nodes_of_trees(n, grammar)
        if not is_builtin_class_instance(n):
            assert isinstance(n, TreeNode)
        return n

    return g


def initialize_object(constructor: Any, args: dict[str, Any]):
    """Calls the constructor, but also saves meta-data."""
    o = constructor(**args)
    o.gengy_init_values = list(args.values())
    return o


@relabel
def random_tree(random: RandomSource, grammar: Grammar, starting_symbol: type[Any], decider: SynthesisDecider):
    """Generates a Random Tree."""

    if starting_symbol is int:
        return decider.random_int()
    elif starting_symbol is float:
        return decider.random_float()
    elif starting_symbol is bool:
        return decider.random_bool()
    elif starting_symbol is str:
        return decider.random_str()
    elif is_generic_tuple(starting_symbol):
        return decider.random_tuple(get_generic_parameters(starting_symbol))
    elif is_generic_list(starting_symbol):
        return decider.random_list(get_generic_parameter(starting_symbol))
    elif is_metahandler(starting_symbol):
        metahandler = starting_symbol.__metadata__[0]
        base_type = get_generic_parameter(starting_symbol)
        # TODO
        return random_tree(random=random, grammar=grammar, starting_symbol=base_type, decider=decider)
    else:

        if starting_symbol not in grammar.all_nodes:
            raise GeneticEngineError(
                f"Symbol {starting_symbol} not in grammar rules.",
            )
        elif starting_symbol in grammar.alternatives:  # Alternatives
            compatible_productions = grammar.alternatives[starting_symbol]
            production = decider.choose_alternatives(compatible_productions)
            return random_tree(random=random, grammar=grammar, starting_symbol=production, decider=decider)
        else:
            args_to_create = {}
            for argn, argt in get_arguments(starting_symbol):
                args_to_create[argn] = random_tree(
                    random=random, grammar=grammar, starting_symbol=argt, decider=decider,
                )
            return initialize_object(starting_symbol, args_to_create)


def random_node(
    r: RandomSource,
    g: Grammar,
    max_depth: int,
    starting_symbol: type[Any] | None = None,
    method: InitializationMethodType = grow_method,
):
    starting_symbol = starting_symbol if starting_symbol else g.starting_symbol
    return method(r, g, max_depth, starting_symbol)


def random_individual(
    r: RandomSource,
    g: Grammar,
    max_depth: int = 5,
    method: InitializationMethodType = grow_method,
) -> TreeNode:
    try:
        assert max_depth >= g.get_min_tree_depth()
    except AssertionError:
        if g.get_min_tree_depth() == 1000000:
            raise GeneticEngineError(
                f"""Grammar's minimal tree depth is {g.get_min_tree_depth()}, which is the default tree depth.
                 It's highly like that there are nodes of your grammar than cannot reach any terminal.""",
            )
        raise GeneticEngineError(
            f"""Cannot use complete grammar for individual creation. Max depth ({max_depth})
            is smaller than grammar's minimal tree depth ({g.get_min_tree_depth()}).""",
        )
    ind = random_node(r, g, max_depth, g.starting_symbol, method)
    assert isinstance(ind, TreeNode)
    return ind


def mutate_inner(
    r: RandomSource,
    g: Grammar,
    i: TreeNode,
    max_depth: int,
    ty: type,
    force_mutate: bool,
    depth_aware_mut: bool,
) -> TreeNode:
    counter = i.gengy_weighted_nodes if depth_aware_mut else i.gengy_nodes
    if counter > 0:
        c = r.randint(0, counter - 1)
        if c == 0 or (c <= i.gengy_distance_to_term and depth_aware_mut) or force_mutate:
            # If Metahandler mutation exists, the mutation process is different
            if any(has_annotated_mutation(arg[1]) for arg in get_arguments(i)):
                options = [(kdx, arg[1]) for kdx, arg in enumerate(get_arguments(i)) if has_annotated_mutation(arg[1])]
                index = r.randint(0, len(options) - 1)
                (index, arg_to_be_mutated) = options[index]

                args = list(i.gengy_init_values)
                args[index] = arg_to_be_mutated.__metadata__[0].mutate(  # type: ignore
                    r,
                    g,
                    random_node,
                    max_depth - 1,
                    get_generic_parameter(arg_to_be_mutated),
                    current_node=args[index],
                )
                mk = mk_save_init(type(i), lambda x: x)(*args)
                return mk

            replacement = None
            for _ in range(5):
                try:
                    replacement = random_node(r, g, max_depth, ty)
                    if replacement != i:
                        break
                except GeneticEngineError:
                    pass
            return replacement if replacement else i
        else:
            if is_abstract(ty) and g.expansion_depthing:
                max_depth -= g.abstract_dist_to_t[ty][type(i)]
            max_depth -= 1
            args = list(i.gengy_init_values)
            c -= i.gengy_distance_to_term if depth_aware_mut else 1
            for idx, (_, field_type) in enumerate(get_arguments(i)):
                child = args[idx]
                if hasattr(child, "gengy_nodes"):
                    count = child.gengy_weighted_nodes if depth_aware_mut else child.gengy_nodes
                    if c <= count:
                        mi = mutate_inner(
                            r,
                            g,
                            child,
                            max_depth,
                            field_type,
                            force_mutate,
                            depth_aware_mut,
                        )
                        args[idx] = mi
                        break
                    else:
                        c -= count
            mk = mk_save_init(i, lambda x: x)(*args)
            return mk
    else:
        rn = None
        for _ in range(5):
            try:
                rn = random_node(r, g, max_depth, ty)
                if rn != i:
                    break
            except GeneticEngineError:
                pass
        return rn if rn else i


def mutate_specific_type_inner(
    r: RandomSource,
    g: Grammar,
    i: TreeNode,
    max_depth: int,
    ty: type,
    specific_type: type,
    n: int,
    depth_aware_mut: bool,
) -> TreeNode:
    if n == 1 and type(i) == specific_type:
        return mutate_inner(
            r,
            g,
            i,
            max_depth,
            ty,
            force_mutate=True,
            depth_aware_mut=depth_aware_mut,
        )
    else:
        args = list(i.gengy_init_values)
        for idx, (_, field_type) in enumerate(get_arguments(i)):
            child = args[idx]
            if hasattr(child, "gengy_nodes"):
                n_options = len(
                    list(find_in_tree_exact(g, specific_type, child, max_depth)),
                )
                if n_options <= n:
                    args[idx] = mutate_specific_type_inner(
                        r,
                        g,
                        child,
                        max_depth,
                        ty,
                        specific_type,
                        n,
                        depth_aware_mut,
                    )
                else:
                    n -= n_options
        return mk_save_init(i, lambda x: x)(*args)


def mutate_specific_type(
    r: RandomSource,
    g: Grammar,
    i: TreeNode,
    max_depth: int,
    target_type: type,
    specific_type: type,
    depth_aware_mut: bool,
) -> TreeNode:
    ch = r.randint(0, 2)
    n_options = len(list(find_in_tree_exact(g, specific_type, i, max_depth)))
    if ch == 0 or n_options == 0:
        new_tree = mutate_inner(
            r,
            g,
            i,
            max_depth,
            target_type,
            force_mutate=False,
            depth_aware_mut=depth_aware_mut,
        )
        relabeled_new_tree = relabel_nodes_of_trees(new_tree, g)
        return relabeled_new_tree
    else:
        n = r.randint(1, n_options)
        new_tree = mutate_specific_type_inner(
            r,
            g,
            i,
            max_depth,
            target_type,
            specific_type,
            n,
            depth_aware_mut,
        )
        relabeled_new_tree = relabel_nodes_of_trees(new_tree, g)
        return relabeled_new_tree


def tree_mutate(
    r: RandomSource,
    g: Grammar,
    i: TreeNode,
    max_depth: int,
    target_type: type,
    depth_aware_mut: bool = False,
) -> Any:
    new_tree = mutate_inner(r, g, i, max_depth, target_type, False, depth_aware_mut)
    relabeled_new_tree = relabel_nodes_of_trees(new_tree, g)
    return relabeled_new_tree


def find_in_tree(g: Grammar, ty: type, o: TreeNode, max_depth: int):
    is_abs = is_abstract(ty)
    if hasattr(o, "gengy_types_this_way"):
        for t in o.gengy_types_this_way:

            def is_valid(node):
                _, depth, _, _ = relabel_nodes(node, g)

                if is_abs and g.expansion_depthing:
                    depth += g.abstract_dist_to_t[ty][t]

                return depth <= max_depth

            if ty in t.__bases__:
                vals = o.gengy_types_this_way[t]
                if vals:
                    yield from filter(is_valid, vals)


def find_in_tree_exact(g: Grammar, ty: type, o: TreeNode, max_depth: int):
    if hasattr(o, "gengy_types_this_way") and ty in o.gengy_types_this_way:
        vals = o.gengy_types_this_way[ty]
        if vals:

            def is_valid(node):
                _, depth, _, _ = relabel_nodes(node, g)
                return depth <= max_depth

            yield from filter(is_valid, vals)


def crossover_inner(
    r: RandomSource,
    g: Grammar,
    i: TreeNode,
    o: TreeNode,
    max_depth: int,
    ty: type,
    force_crossover: bool,
    depth_aware_co: bool,
) -> Any:
    counter = i.gengy_weighted_nodes if depth_aware_co else i.gengy_nodes
    if counter > 0:
        c = r.randint(0, counter - 1)
        if c == 0 or (c <= i.gengy_distance_to_term and depth_aware_co) or force_crossover:
            replacement = None
            args_with_specific_crossover = [has_annotated_crossover(arg[1]) for arg in get_arguments(i)]
            if any(args_with_specific_crossover):
                crossover_possibilities = len(args_with_specific_crossover)
                crossover_choice = r.randint(
                    0,
                    crossover_possibilities - 1,
                )
                options = list(find_in_tree_exact(g, type(i), o, max_depth))
                if not options:
                    pass  # Replace whole node
                else:
                    (index, arg_to_be_crossovered) = [(kdx, arg) for kdx, arg in enumerate(get_arguments(i))][
                        crossover_choice
                    ]
                    args = list(i.gengy_init_values)
                    if has_annotated_crossover(arg_to_be_crossovered[1]):
                        args[index] = (
                            arg_to_be_crossovered[1]
                            .__metadata__[0]  # type: ignore
                            .crossover(
                                r,
                                g,
                                options,
                                arg_to_be_crossovered[0],
                                ty,
                                current_node=args[index],
                            )
                        )
                        return mk_save_init(type(i), lambda x: x)(*args)

            options = list(find_in_tree(g, ty, o, max_depth))
            if options:
                replacement = r.choice(options)
            if replacement is None:
                for _ in range(5):
                    replacement = random_node(r, g, max_depth, ty)
                    if replacement != i:
                        break

            return replacement
        else:
            if is_abstract(ty) and g.expansion_depthing:
                max_depth -= g.abstract_dist_to_t[ty][type(i)]
            max_depth -= 1
            args = list(i.gengy_init_values)
            c -= i.gengy_distance_to_term if depth_aware_co else 1
            for idx, (field, field_type) in enumerate(get_arguments(i)):
                child = args[idx]
                if hasattr(child, "gengy_nodes"):
                    count = child.gengy_weighted_nodes if depth_aware_co else child.gengy_nodes
                    if c <= count:
                        args[idx] = crossover_inner(
                            r,
                            g,
                            child,
                            o,
                            max_depth,
                            field_type,
                            force_crossover=False,
                            depth_aware_co=depth_aware_co,
                        )
                        break
                    else:
                        c -= count
            return mk_save_init(i, lambda x: x)(*args)
    else:
        return i


def crossover_specific_type_inner(
    r: RandomSource,
    g: Grammar,
    i: TreeNode,
    o: TreeNode,
    max_depth: int,
    ty: type,
    specific_type: type,
    n: int,
    depth_aware_co: bool,
) -> TreeNode:
    if n == 1 and type(i) == specific_type:
        return crossover_inner(
            r,
            g,
            i,
            o,
            max_depth,
            ty,
            force_crossover=True,
            depth_aware_co=depth_aware_co,
        )
    else:
        args = list(i.gengy_init_values)
        for idx, (_, field_type) in enumerate(get_arguments(i)):
            child = args[idx]
            n_options = len(
                list(find_in_tree_exact(g, specific_type, child, max_depth)),
            )
            if n_options <= n:
                args[idx] = crossover_specific_type_inner(
                    r,
                    g,
                    child,
                    o,
                    max_depth,
                    ty,
                    specific_type,
                    n,
                    depth_aware_co=depth_aware_co,
                )
            else:
                n -= n_options
        return mk_save_init(i, lambda x: x)(*args)


def crossover_specific_type(
    r: RandomSource,
    g: Grammar,
    i: TreeNode,
    o: TreeNode,
    max_depth: int,
    target_type: type,
    specific_type: type,
    depth_aware_co: bool,
) -> TreeNode:
    ch = r.randint(0, 1)
    n_options_i = len(list(find_in_tree_exact(g, specific_type, i, max_depth)))
    n_options_o = len(list(find_in_tree_exact(g, specific_type, o, max_depth)))
    if ch == 0 or n_options_i == 0 or n_options_o == 0:
        new_tree = crossover_inner(
            r,
            g,
            i,
            o,
            max_depth,
            target_type,
            force_crossover=False,
            depth_aware_co=depth_aware_co,
        )
        relabeled_new_tree = relabel_nodes_of_trees(new_tree, g)
        return relabeled_new_tree
    else:
        n = r.randint(1, n_options_i)
        new_tree = crossover_specific_type_inner(
            r,
            g,
            i,
            o,
            max_depth,
            target_type,
            specific_type,
            n,
            depth_aware_co=depth_aware_co,
        )
        relabeled_new_tree = relabel_nodes_of_trees(new_tree, g)
        return relabeled_new_tree


def tree_crossover(
    r: RandomSource,
    g: Grammar,
    p1: TreeNode,
    p2: TreeNode,
    max_depth: int,
    specific_type: type | None = None,
    depth_aware_co: bool = False,
) -> tuple[TreeNode, TreeNode]:
    """Given the two input trees [p1] and [p2], the grammar and the random
    source, this function returns two trees that are created by crossing over.

    [p1] and [p2].

    The first tree returned has [p1] as the base, and the second tree
    has [p2] as a base.
    """
    if specific_type:
        new_tree1 = crossover_specific_type(
            r,
            g,
            p1,
            p2,
            max_depth,
            g.starting_symbol,
            specific_type,
            depth_aware_co=depth_aware_co,
        )
    else:
        new_tree1 = crossover_inner(
            r,
            g,
            p1,
            p2,
            max_depth,
            g.starting_symbol,
            force_crossover=False,
            depth_aware_co=depth_aware_co,
        )
    relabeled_new_tree1 = relabel_nodes_of_trees(new_tree1, g)

    if specific_type:
        new_tree2 = crossover_specific_type(
            r,
            g,
            p2,
            p1,
            max_depth,
            g.starting_symbol,
            specific_type,
            depth_aware_co=depth_aware_co,
        )
    else:
        new_tree2 = crossover_inner(
            r,
            g,
            p2,
            p1,
            max_depth,
            g.starting_symbol,
            force_crossover=False,
            depth_aware_co=depth_aware_co,
        )
    relabeled_new_tree2 = relabel_nodes_of_trees(new_tree2, g)
    return relabeled_new_tree1, relabeled_new_tree2


class TreeBasedRepresentation(
    Representation[TreeNode, TreeNode],
    RepresentationWithMutation[TreeNode],
    RepresentationWithCrossover[TreeNode],
):
    """This class represents the tree representation of an individual.

    In this approach, the genotype and the phenotype are exactly the
    same.
    """

    def __init__(
        self,
        grammar: Grammar,
        max_depth: int,
        initialization_method: InitializationMethodType = pi_grow_method,
    ):
        self.grammar = grammar
        self.max_depth = max_depth
        self.initialization_method = initialization_method

    def create_genotype(self, random: RandomSource, **kwargs) -> TreeNode:
        return random_tree(
            random=random,
            grammar=self.grammar,
            starting_symbol=self.grammar.starting_symbol,
            decider=BasicSynthesisDecider(random, self.grammar),
        )

    def genotype_to_phenotype(self, genotype: TreeNode) -> TreeNode:
        return genotype

    def mutate(self, random: RandomSource, internal: TreeNode, **kwargs) -> TreeNode:
        return tree_mutate(
            random,
            self.grammar,
            internal,
            max_depth=self.max_depth,
            target_type=self.grammar.starting_symbol,
        )

    def crossover(
        self,
        random: RandomSource,
        parent1: TreeNode,
        parent2: TreeNode,
        **kwargs,
    ) -> tuple[TreeNode, TreeNode]:
        return tree_crossover(random, self.grammar, parent1, parent2, self.max_depth)
