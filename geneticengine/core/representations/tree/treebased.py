from __future__ import annotations

import sys
from copy import deepcopy
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TypeVar

import z3

from geneticengine.core.decorators import get_gengy
from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import Representation
from geneticengine.core.representations.tree.utils import GengyList
from geneticengine.core.representations.tree.utils import relabel_nodes
from geneticengine.core.representations.tree.utils import relabel_nodes_of_trees
from geneticengine.core.tree import TreeNode
from geneticengine.core.utils import build_finalizers
from geneticengine.core.utils import get_arguments
from geneticengine.core.utils import get_generic_parameter
from geneticengine.core.utils import has_annotated_crossover
from geneticengine.core.utils import has_annotated_mutation
from geneticengine.core.utils import is_abstract
from geneticengine.core.utils import is_annotated
from geneticengine.core.utils import is_generic_list
from geneticengine.core.utils import strip_annotations
from geneticengine.exceptions import GeneticEngineError
from geneticengine.metahandlers.base import is_metahandler

T = TypeVar("T")


def apply_metahandler(
    r: Source,
    g: Grammar,
    receiver,
    new_symbol,
    depth: int,
    ty: type[Any],
    context: dict[str, str],
) -> Any:
    """
    This method applies a metahandler to use a custom generator for things of a given type.
    As an example, AnnotatedType[int, IntRange(3,10)] will use the IntRange.generate(r, recursive_generator)
    The generator is the annotation on the type ("__metadata__").
    """
    metahandler = ty.__metadata__[0]
    base_type = get_generic_parameter(ty)
    return metahandler.generate(
        r,
        g,
        receiver,
        new_symbol,
        depth,
        base_type,
        context,
    )  # todo: last argument


# TODO: make non static
class SMTResolver:
    clauses: list[Any] = []
    receivers: dict[str, Callable] = {}
    types: dict[str, Callable] = {}

    @staticmethod
    def add_clause(claus, recs: dict[str, Callable]):
        SMTResolver.clauses.extend(claus)
        for k, v in recs.items():
            SMTResolver.receivers[k] = v

    @staticmethod
    def register_type(name, typ):
        SMTResolver.types[name] = SMTResolver.to_z3_typ(typ)

    @staticmethod
    def to_z3_typ(typ):
        return z3.Bool if typ == bool else z3.Int if typ == int else z3.Real

    @staticmethod
    def resolve_clauses():

        if not SMTResolver.receivers:
            return  # don't try to smt solve if we don't need to

        solver = z3.Solver()

        solver.set(":random-seed", 1)
        solver.reset()

        for clause in SMTResolver.clauses:
            solver.add(clause(SMTResolver.types))
        res = solver.check()

        if res != z3.sat:
            raise Exception(f"{solver} failed with {res}")

        model = solver.model()
        for (name, recv) in SMTResolver.receivers.items():
            evaled = model.eval(
                SMTResolver.types[name](
                    name,
                ),
                model_completion=True,
            )

            recv(SMTResolver.get_type(evaled))

        SMTResolver.clauses = []
        SMTResolver.receivers = {}
        SMTResolver.types = {}

    @staticmethod
    def get_type(evaled):
        if type(evaled) == z3.z3.BoolRef:
            evaled = bool(str(evaled))
        elif type(evaled) == z3.z3.IntNumRef:
            evaled = int(str(evaled))
        elif type(evaled) == z3.z3.RatNumRef:
            evaled = eval(str(evaled))
        else:
            raise NotImplementedError(
                f"Don't know what to do with {type(evaled)} {evaled}",
            )
        return evaled

    @staticmethod
    def register_const(ident, val):
        SMTResolver.register_type(ident, type(val))
        ty = SMTResolver.types[ident]
        SMTResolver.clauses.append(lambda _: ty(ident) == val)


def Grow(
    r: Source,
    g: Grammar,
    depth: int,
    starting_symbol: type[Any] = int,
):
    def filter_choices(possible_choices: list[type], depth):
        valid_productions = [
            vp for vp in possible_choices if g.get_distance_to_terminal(vp) <= depth
        ]
        return valid_productions

    def handle_symbol(
        next_type,
        next_finalizer,
        depth: int,
        ident: str,
        ctx: dict[str, str],
    ):
        expand_node(
            r,
            g,
            handle_symbol,
            filter_choices,
            next_finalizer,
            depth,
            next_type,
            ident,
            ctx,
        )

    state = {}

    def final_finalize(x):
        state["final"] = x

    handle_symbol(starting_symbol, final_finalize, depth, "root", ctx={})
    SMTResolver.resolve_clauses()
    n = state["final"]
    relabel_nodes_of_trees(n, g)
    return n


def PI_Grow(
    r: Source,
    g: Grammar,
    depth: int,
    starting_symbol: type[Any] = int,
):
    state = {}

    def final_finalize(x):
        state["final"] = x

    prodqueue = []
    nRecs = [0]

    def handle_symbol(
        next_type,
        next_finalizer,
        depth: int,
        ident: str,
        ctx: dict[str, str],
    ):
        prodqueue.append((next_type, next_finalizer, depth, ident, ctx))
        if next_type in g.recursive_prods:
            nRecs[0] += 1

    handle_symbol(starting_symbol, final_finalize, depth, "root", ctx={})

    def filter_choices(possible_choices: list[type], depth):
        valid_productions = [
            vp for vp in possible_choices if g.distanceToTerminal[vp] <= depth
        ]
        if (nRecs[0] == 0) and any(  # Are we the last recursive symbol?
            [
                prod in g.recursive_prods for prod in valid_productions
            ],  # Are there any  recursive symbols in our expansion?
        ):
            valid_productions = [
                vp for vp in valid_productions if vp in g.recursive_prods
            ]  # If so, then only expand into recursive symbols

        return valid_productions

    while prodqueue:
        next_type, next_finalizer, depth, ident, ctx = r.pop_random(prodqueue)
        if next_type in g.recursive_prods:
            nRecs[0] -= 1
        expand_node(
            r,
            g,
            handle_symbol,
            filter_choices,
            next_finalizer,
            depth,
            next_type,
            ident,
            ctx,
        )
    SMTResolver.resolve_clauses()
    n = state["final"]
    relabel_nodes_of_trees(n, g)
    return n


def random_node(
    r: Source,
    g: Grammar,
    max_depth: int,
    starting_symbol: type[Any] = None,
    method=PI_Grow,
):
    if starting_symbol is None:
        starting_symbol = g.starting_symbol
    return method(r, g, max_depth, starting_symbol)


def mk_save_init(starting_symbol: Any, receiver: Callable):
    if isinstance(starting_symbol, type):
        pass
    elif isinstance(starting_symbol, GengyList):
        starting_symbol = starting_symbol.new_like
    else:
        starting_symbol = type(starting_symbol)

    def fin_recv(*x):
        built = starting_symbol(*x)
        built.gengy_init_values = x
        return receiver(built)

    return fin_recv


def expand_node(
    r: Source,
    g: Grammar,
    new_symbol,  # Method to find new_symbol (?)
    filter_choices,
    receiver,
    depth,
    starting_symbol: type,
    id: str,
    ctx: dict[str, str],
) -> Any:
    """
    Creates a random node of a given type (starting_symbol)
    """
    if depth < 0:
        raise GeneticEngineError("Recursion Depth reached")
    if depth < g.get_distance_to_terminal(starting_symbol):
        raise GeneticEngineError(
            "There will be no depth sufficient for symbol {} in this grammar (provided: {}, required: {}).".format(
                starting_symbol,
                depth,
                g.get_distance_to_terminal(
                    starting_symbol,
                ),
            ),
        )

    if starting_symbol is int:
        val = r.randint(0, 100, str(starting_symbol))
        SMTResolver.register_const(id, val)
        receiver(val)
        return
    elif starting_symbol is float:
        valf = r.random_float(-100, 100, str(starting_symbol))
        SMTResolver.register_const(id, valf)
        receiver(valf)
        return
    elif starting_symbol is bool:
        valb = r.random_bool(str(starting_symbol))
        SMTResolver.register_const(id, valb)
        receiver(valb)
        return
    elif is_generic_list(starting_symbol):
        ctx = ctx.copy()
        ctx["_"] = id
        r.random_list(
            receiver,
            new_symbol,
            depth,
            starting_symbol,
            ctx,
            str(starting_symbol),
        )
        return
    elif is_metahandler(starting_symbol):
        ctx = ctx.copy()
        ctx["_"] = id
        apply_metahandler(
            r,
            g,
            receiver,
            new_symbol,
            depth,
            starting_symbol,
            ctx,
        )
        return
    else:
        if starting_symbol not in g.all_nodes:
            raise GeneticEngineError(
                f"Symbol {starting_symbol} not in grammar rules.",
            )

        if starting_symbol in g.alternatives:  # Alternatives
            extra_depth = 0
            if is_abstract(starting_symbol) and g.expansion_depthing:
                extra_depth = 1

            compatible_productions = g.alternatives[starting_symbol]
            valid_productions = filter_choices(
                compatible_productions,
                depth - extra_depth,
            )
            if not valid_productions:
                raise GeneticEngineError(
                    "No productions for non-terminal node with type: {} in depth {} (minimum required: {}).".format(
                        starting_symbol,
                        depth - extra_depth,
                        str(
                            [
                                (vp, g.distanceToTerminal[vp])
                                for vp in compatible_productions
                            ],
                        ),
                    ),
                )
            if any(["weight" in get_gengy(p) for p in valid_productions]):
                weights = [get_gengy(p).get("weight", 1.0) for p in valid_productions]
                rule = r.choice_weighted(
                    valid_productions,
                    weights,
                    str(starting_symbol),
                )
            else:
                rule = r.choice(valid_productions, str(starting_symbol))
            new_symbol(rule, receiver, depth - extra_depth, id, ctx)
        else:  # Normal production
            args = get_arguments(starting_symbol)
            ctx = ctx.copy()
            l: list[Any] = []
            for argn, _ in args:
                name = id + "_" + argn
                ctx[argn] = name

                def fn(val, name=name):
                    pass

                l.append(fn)

            fins = build_finalizers(
                mk_save_init(starting_symbol, receiver),
                len(args),
                l,
            )
            for i, (argn, argt) in enumerate(args):
                new_symbol(argt, fins[i], depth - 1, id + "_" + argn, ctx)


def random_individual(r: Source, g: Grammar, max_depth: int = 5) -> TreeNode:
    try:
        assert max_depth >= g.get_min_tree_depth()
    except:
        if g.get_min_tree_depth() == 1000000:
            raise GeneticEngineError(
                f"Grammar's minimal tree depth is {g.get_min_tree_depth()}, which is the default tree depth. It's highly like that there are nodes of your grammar than cannot reach any terminal.",
            )
        raise GeneticEngineError(
            f"Cannot use complete grammar for individual creation. Max depth ({max_depth}) is smaller than grammar's minimal tree depth ({g.get_min_tree_depth()}).",
        )
    ind = random_node(r, g, max_depth, g.starting_symbol)
    assert isinstance(ind, TreeNode)
    return ind


def mutate_inner(
    r: Source,
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
        if (
            c == 0
            or (c <= i.gengy_distance_to_term and depth_aware_mut)
            or force_mutate
        ):
            # If Metahandler mutation exists, the mutation process is different
            args_with_specific_mutation = [
                has_annotated_mutation(arg[1]) for arg in get_arguments(i)
            ]
            if any(args_with_specific_mutation):
                mutation_possibilities = len(args_with_specific_mutation)
                mutation_choice = r.randint(
                    0,
                    mutation_possibilities - 1,
                )
                (index, arg_to_be_mutated) = [
                    (kdx, arg[1])
                    for kdx, arg in enumerate(get_arguments(i))
                    if args_with_specific_mutation[kdx]
                ][mutation_choice]
                args = list(i.gengy_init_values)
                args[index] = arg_to_be_mutated.__metadata__[0].mutate(  # type: ignore
                    r,
                    g,
                    random_node,
                    max_depth - 1,
                    get_generic_parameter(arg_to_be_mutated),
                    method=Grow,
                    current_node=args[index],
                )
                mk = mk_save_init(type(i), lambda x: x)(*args)
                return mk

            for _ in range(5):
                replacement = random_node(r, g, max_depth, ty, method=Grow)
                if replacement != i:
                    break
            return replacement
        else:
            if is_abstract(ty) and g.expansion_depthing:
                max_depth -= g.abstract_dist_to_t[ty][type(i)]
            max_depth -= 1
            args = list(i.gengy_init_values)
            c -= i.gengy_distance_to_term if depth_aware_mut else 1
            for idx, (_, field_type) in enumerate(get_arguments(i)):
                child = args[idx]
                if hasattr(child, "gengy_nodes"):
                    count = (
                        child.gengy_weighted_nodes
                        if depth_aware_mut
                        else child.gengy_nodes
                    )
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
        for _ in range(5):
            rn = random_node(r, g, max_depth, ty, method=Grow)
            if rn != i:
                break
        return rn


def mutate_specific_type_inner(
    r: Source,
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
    r: Source,
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


def mutate(
    r: Source,
    g: Grammar,
    i: TreeNode,
    max_depth: int,
    target_type: type,
    specific_type: type | None = None,
    depth_aware_mut: bool = False,
) -> Any:
    if specific_type:
        return mutate_specific_type(
            r,
            g,
            i,
            max_depth,
            target_type,
            specific_type,
            depth_aware_mut,
        )
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
    if hasattr(o, "gengy_types_this_way"):

        vals = o.gengy_types_this_way[ty]
        if vals:

            def is_valid(node):
                _, depth, _, _ = relabel_nodes(node, g)
                return depth <= max_depth

            yield from filter(is_valid, vals)


def crossover_inner(
    r: Source,
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
        if (
            c == 0
            or (c <= i.gengy_distance_to_term and depth_aware_co)
            or force_crossover
        ):
            replacement = None
            args_with_specific_crossover = [
                has_annotated_crossover(arg[1]) for arg in get_arguments(i)
            ]
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
                    (index, arg_to_be_crossovered) = [
                        (kdx, arg)
                        for kdx, arg in enumerate(get_arguments(i))
                        if args_with_specific_crossover[kdx]
                    ][crossover_choice]
                    args = list(i.gengy_init_values)
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
                    replacement = random_node(r, g, max_depth, ty, method=Grow)
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
                    count = (
                        child.gengy_weighted_nodes
                        if depth_aware_co
                        else child.gengy_nodes
                    )
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
    r: Source,
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
    r: Source,
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


def crossover(
    r: Source,
    g: Grammar,
    p1: TreeNode,
    p2: TreeNode,
    max_depth: int,
    specific_type: type | None = None,
    depth_aware_co: bool = False,
) -> tuple[TreeNode, TreeNode]:
    """
    Given the two input trees [p1] and [p2], the grammar and the random source, this function returns two trees that are created by crossing over [p1] and [p2]. The first tree returned has [p1] as the base, and the second tree has [p2] as a base.
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


class TreeBasedRepresentation(Representation[TreeNode]):
    def create_individual(self, r: Source, g: Grammar, depth: int) -> TreeNode:
        return random_individual(r, g, depth)

    def mutate_individual(
        self,
        r: Source,
        g: Grammar,
        ind: TreeNode,
        depth: int,
        ty: type,
        specific_type: type = None,
        depth_aware_mut: bool = False,
    ) -> TreeNode:
        new_ind = mutate(r, g, ind, depth, ty, specific_type, depth_aware_mut)
        return new_ind

    def crossover_individuals(
        self,
        r: Source,
        g: Grammar,
        i1: TreeNode,
        i2: TreeNode,
        max_depth: int,
        specific_type: type = None,
        depth_aware_co: bool = False,
    ) -> tuple[TreeNode, TreeNode]:
        return crossover(r, g, i1, i2, max_depth, specific_type, depth_aware_co)

    def genotype_to_phenotype(self, g: Grammar, genotype: TreeNode) -> TreeNode:
        return genotype


treebased_representation = TreeBasedRepresentation()
