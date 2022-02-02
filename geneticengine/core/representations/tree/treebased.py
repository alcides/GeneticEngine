import sys
from copy import deepcopy

from typing import (
    Any,
    Dict,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Tuple,
    List,
    Callable,
)

import z3

from geneticengine.core.decorators import get_gengy
from geneticengine.core.random.sources import Source
from geneticengine.core.grammar import Grammar
from geneticengine.core.representations.api import Representation
from geneticengine.core.representations.tree.utils import relabel_nodes_of_trees

from geneticengine.core.tree import TreeNode
from geneticengine.core.utils import (
    get_arguments,
    is_generic_list,
    get_generic_parameter,
    build_finalizers,
)
from geneticengine.exceptions import GeneticEngineError
from geneticengine.metahandlers.base import is_metahandler


def random_bool(r: Source) -> int:
    return r.choice([True, False])


def random_int(r: Source) -> int:
    return r.randint(-(sys.maxsize - 1), sys.maxsize)


def random_float(r: Source) -> float:
    return r.random_float(-100, 100)


T = TypeVar("T")


def random_list(
    r: Source, receiver, new_symbol, depth: int, ty: Type[List[T]], ctx: Dict[str, str]
):
    inner_type = get_generic_parameter(ty)
    size = r.randint(0, depth - 1)
    fins = build_finalizers(lambda *x: receiver(list(x)), size)
    ident = ctx["_"]
    for i, fin in enumerate(fins):
        nctx = ctx.copy()
        nident = ident + "_" + str(i)
        nctx["_"] = nident
        new_symbol(inner_type, fin, depth - 1, nident, nctx)


def apply_metahandler(
    r: Source,
    g: Grammar,
    receiver,
    new_symbol,
    depth: int,
    ty: Type[Any],
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
        r, g, receiver, new_symbol, depth, base_type, context
    )  # todo: last argument


# TODO: make non static
class SMTResolver(object):
    clauses: List[Any] = []
    receivers: dict[str, Callable] = {}
    types: dict[str, Callable] = {}

    @staticmethod
    def add_clause(claus, recs: dict[str, Callable]):
        SMTResolver.clauses.extend(claus)
        # print(recs)
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
            evaled = model.eval(SMTResolver.types[name](name), model_completion=True)

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
                f"Don't know what to do with {type(evaled)} {evaled}"
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
    starting_symbol: Type[Any] = int,
):
    def filter_choices(possible_choices: List[type], depth):
        valid_productions = [
            vp for vp in possible_choices if g.get_distance_to_terminal(vp) <= depth
        ]
        return valid_productions

    def handle_symbol(
        next_type, next_finalizer, depth: int, ident: str, ctx: Dict[str, str]
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
    relabel_nodes_of_trees(n, g.non_terminals)
    return n


def PI_Grow(
    r: Source,
    g: Grammar,
    depth: int,
    starting_symbol: Type[Any] = int,
):
    state = {}

    def final_finalize(x):
        state["final"] = x

    prodqueue = []
    nRecs = [0]

    def handle_symbol(
        next_type, next_finalizer, depth: int, ident: str, ctx: Dict[str, str]
    ):
        prodqueue.append((next_type, next_finalizer, depth, ident, ctx))
        if next_type in g.recursive_prods:
            nRecs[0] += 1

    handle_symbol(starting_symbol, final_finalize, depth, "root", ctx={})

    def filter_choices(possible_choices: List[type], depth):
        valid_productions = [
            vp for vp in possible_choices if g.distanceToTerminal[vp] <= depth
        ]
        if (nRecs[0] == 0) and any(  # Are we the last recursive symbol?
            [
                prod in g.recursive_prods for prod in valid_productions
            ]  # Are there any  recursive symbols in our expansion?
        ):
            valid_productions = [
                vp for vp in valid_productions if vp in g.recursive_prods
            ]  # If so, then only expand into recursive symbols

        return valid_productions

    while prodqueue:
        # print(nRecs[0])
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
    relabel_nodes_of_trees(n, g.non_terminals)
    return n


def random_node(
    r: Source,
    g: Grammar,
    max_depth: int,
    starting_symbol: Type[Any] = int,
    method=PI_Grow,
):
    return method(r, g, max_depth, starting_symbol)


def expand_node(
    r: Source,
    g: Grammar,
    new_symbol,
    filter_choices,
    receiver,
    depth,
    starting_symbol,
    id: str,
    ctx: Dict[str, str],
) -> Any:
    """
    Creates a random node of a given type (starting_symbol)
    """
    if depth < 0:
        raise GeneticEngineError("Recursion Depth reached")
    if depth < g.get_distance_to_terminal(starting_symbol):
        raise GeneticEngineError(
            "There will be no depth sufficient for symbol {} in this grammar (provided: {}, required: {}).".format(
                starting_symbol, depth, g.get_distance_to_terminal(starting_symbol)
            )
        )

    if starting_symbol is int:
        val = random_int(r)
        SMTResolver.register_const(id, val)
        receiver(val)
        return
    elif starting_symbol is float:
        valf = random_float(r)
        SMTResolver.register_const(id, valf)
        receiver(valf)
        return
    elif is_generic_list(starting_symbol):
        ctx = ctx.copy()
        ctx["_"] = id
        random_list(r, receiver, new_symbol, depth, starting_symbol, ctx)
        return
    elif is_metahandler(starting_symbol):
        ctx = ctx.copy()
        ctx["_"] = id
        apply_metahandler(r, g, receiver, new_symbol, depth, starting_symbol, ctx)
        return
    else:
        if starting_symbol not in g.all_nodes:
            raise GeneticEngineError(f"Symbol {starting_symbol} not in grammar rules.")

        if starting_symbol in g.alternatives:  # Alternatives
            compatible_productions = g.alternatives[starting_symbol]
            valid_productions = filter_choices(compatible_productions, depth - 1)
            if not valid_productions:
                raise GeneticEngineError(
                    "No productions for non-terminal node with type: {} in depth {} (minimum required: {}).".format(
                        starting_symbol,
                        depth,
                        str(
                            [
                                (vp, g.distanceToTerminal[vp])
                                for vp in compatible_productions
                            ]
                        ),
                    )
                )
            if any(["weight" in get_gengy(p) for p in valid_productions]):
                weights = [get_gengy(p).get("weight", 1.0) for p in valid_productions]
                rule = r.choice_weighted(valid_productions, weights)
            else:
                rule = r.choice(valid_productions)
            new_symbol(rule, receiver, depth - 1, id, ctx)
        else:  # Normal production
            args = get_arguments(starting_symbol)
            ctx = ctx.copy()
            l: List[Any] = []
            for argn, _ in args:
                name = id + "_" + argn
                ctx[argn] = name

                def fn(val, name=name):
                    pass
                    # print(f"{name}={val}")
                    # SMTResolver.register_const(name, val)

                l.append(fn)

            fins = build_finalizers(
                lambda *x: receiver(starting_symbol(*x)), len(args), l
            )
            for i, (argn, argt) in enumerate(args):
                new_symbol(argt, fins[i], depth - 1, id + "_" + argn, ctx)


def random_individual(r: Source, g: Grammar, max_depth: int = 5) -> TreeNode:
    try:
        assert max_depth >= g.get_min_tree_depth()
    except:
        if g.get_min_tree_depth() == 1000000:
            raise GeneticEngineError(
                f"Grammar's minimal tree depth is {g.get_min_tree_depth()}, which is the default tree depth. It's highly like that there are nodes of your grammar than cannot reach any terminal."
            )
        raise GeneticEngineError(
            f"Cannot use complete grammar for individual creation. Max depth ({max_depth}) is smaller than grammar's minimal tree depth ({g.get_min_tree_depth()})."
        )
    ind = random_node(r, g, max_depth, g.starting_symbol)
    assert isinstance(ind, TreeNode)
    return ind


def mutate_inner(
    r: Source, g: Grammar, i: TreeNode, max_depth: int, ty: Type
) -> TreeNode:
    if i.nodes > 0:
        c = r.randint(0, i.nodes - 1)
        if c == 0:
            try:
                replacement = random_node(
                    r, g, max_depth - i.depth + 1, ty, method=Grow
                )
                return replacement
            except:
                return i
        else:
            for field in i.__annotations__:
                child = getattr(i, field)
                field_type = i.__annotations__[field]
                if hasattr(child, "nodes"):
                    count = child.nodes
                    if c <= count:
                        setattr(
                            i, field, mutate_inner(r, g, child, max_depth, field_type)
                        )
                        return i
                    else:
                        c -= count
            return i
    else:
        return random_node(r, g, max_depth - i.depth + 1, ty, method=Grow)


def mutate(
    r: Source, g: Grammar, i: TreeNode, max_depth: int, target_type: Type
) -> Any:
    new_tree = mutate_inner(r, g, deepcopy(i), max_depth, target_type)
    relabeled_new_tree = relabel_nodes_of_trees(new_tree, g.non_terminals, max_depth)
    return relabeled_new_tree


def find_in_tree(ty: type, o: TreeNode, max_depth: int):
    if ty in o.__class__.__bases__ and o.distance_to_term <= max_depth:
        yield o
    if hasattr(o, "__annotations__"):
        for field in o.__annotations__:
            child = getattr(o, field)
            yield from find_in_tree(ty, child, max_depth)


def tree_crossover_inner(
    r: Source, g: Grammar, i: TreeNode, o: TreeNode, ty: Type, max_depth: int
) -> Any:
    if i.nodes > 0:
        c = r.randint(0, i.nodes - 1)
        if c == 0:
            replacement = None
            options = list(find_in_tree(ty, o, max_depth - i.depth + 1))
            if options:
                replacement = r.choice(options)
            if replacement is None:
                try:
                    replacement = random_node(
                        r, g, max_depth - i.depth + 1, ty, method=Grow
                    )
                except:
                    return i
            return replacement
        else:
            args = {}
            for field in i.__annotations__:
                child = getattr(i, field)
                field_type = i.__annotations__[field]
                if hasattr(child, "nodes"):
                    count = child.nodes
                    if c <= count:
                        args[field] = tree_crossover_inner(
                            r, g, getattr(i, field), o, field_type, max_depth
                        )
                        continue
                    else:
                        c -= count
                        args[field] = child
            return modify_and_construct(i, args)
    else:
        return i


def modify_and_construct(source, mods):
    cons = type(source)
    init = cons.__init__
    var_list = init.__code__.co_varnames[
        1 : init.__code__.co_argcount
    ]  # todo: cursed, we should store the args we pass
    # todo: to the init in a sep variable and use them instead, but oh well
    final = {}
    for var in var_list:
        if var in mods:
            final[var] = mods[var]
        else:
            final[var] = getattr(source, var)
    return cons(**final)


def tree_crossover(
    r: Source, g: Grammar, p1: TreeNode, p2: TreeNode, max_depth: int
) -> Tuple[TreeNode, TreeNode]:
    """
    Given the two input trees [p1] and [p2], the grammar and the random source, this function returns two trees that are created by crossing over [p1] and [p2]. The first tree returned has [p1] as the base, and the second tree has [p2] as a base.
    """
    new_tree1 = tree_crossover_inner(r, g, p1, p2, g.starting_symbol, max_depth)
    relabeled_new_tree1 = relabel_nodes_of_trees(new_tree1, g.non_terminals)
    new_tree2 = tree_crossover_inner(r, g, p2, p1, g.starting_symbol, max_depth)
    relabeled_new_tree2 = relabel_nodes_of_trees(new_tree2, g.non_terminals)
    return relabeled_new_tree1, relabeled_new_tree2


def tree_crossover_single_tree(
    r: Source, g: Grammar, p1: TreeNode, p2: TreeNode, max_depth: int
) -> TreeNode:
    """
    Given the two input trees [p1] and [p2], the grammar and the random source, this function returns one tree that is created by crossing over [p1] and [p2]. The tree returned has [p1] as the base.
    """
    new_tree = tree_crossover_inner(
        r, g, deepcopy(p1), deepcopy(p2), g.starting_symbol, max_depth
    )
    relabeled_new_tree = relabel_nodes_of_trees(new_tree, g.non_terminals)
    return relabeled_new_tree


class TreeBasedRepresentation(Representation[TreeNode]):
    def create_individual(self, r: Source, g: Grammar, depth: int) -> TreeNode:
        return random_individual(r, g, depth)

    def mutate_individual(
        self, r: Source, g: Grammar, ind: TreeNode, depth: int, ty: Type
    ) -> TreeNode:
        return mutate(r, g, ind, depth, ty)

    def crossover_individuals(
        self, r: Source, g: Grammar, i1: TreeNode, i2: TreeNode, max_depth: int
    ) -> Tuple[TreeNode, TreeNode]:
        return tree_crossover(r, g, i1, i2, max_depth)

    def genotype_to_phenotype(self, g: Grammar, genotype: TreeNode) -> TreeNode:
        return genotype


treebased_representation = TreeBasedRepresentation()
