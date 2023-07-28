from __future__ import annotations
import sys
from typing import Any, TypeVar
from typing import Callable

from geneticengine.core.decorators import get_gengy
from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.tree_smt.smt import SMTResolver
from geneticengine.core.representations.tree_smt.utils import GengyList
from geneticengine.core.representations.tree_smt.utils import relabel_nodes_of_trees
from geneticengine.core.utils import build_finalizers
from geneticengine.core.utils import get_arguments
from geneticengine.core.utils import get_generic_parameter
from geneticengine.core.utils import is_abstract
from geneticengine.core.utils import is_builtin_class_instance
from geneticengine.core.utils import is_generic_list
from geneticengine.exceptions import GeneticEngineError
from geneticengine.metahandlers.base import is_metahandler


def apply_metahandler(
    r: Source,
    g: Grammar,
    receiver,
    new_symbol,
    depth: int,
    ty: type[Any],
    context: dict[str, str],
) -> Any:
    """This method applies a metahandler to use a custom generator for things
    of a given type.

    As an example, AnnotatedType[int, IntRange(3,10)] will use the
    IntRange.generate(r, recursive_generator). The generator is the
    annotation on the type ("__metadata__").
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


InitializationMethodType = Callable[[Source, Grammar, int, type[Any]], Any]


def grow_method(
    r: Source,
    g: Grammar,
    depth: int,
    starting_symbol: type[Any] = int,
):
    """Implements the standard Grow tree-initialization method, where trees are
    naturally grown from the grammar."""

    def filter_choices(possible_choices: list[type], depth):
        valid_productions = [vp for vp in possible_choices if g.get_distance_to_terminal(vp) <= depth]
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
    n = state["final"]
    relabel_nodes_of_trees(n, g)
    return n


def full_method(
    r: Source,
    g: Grammar,
    depth: int,
    starting_symbol: type[Any] = int,
):
    """Full tree-initialization method.

    Trees are grown from the grammar with all branches as deep as
    possible, making full trees.
    """

    def filter_choices(possible_choices: list[type], depth):
        valid_productions = [vp for vp in possible_choices if g.get_distance_to_terminal(vp) <= depth]
        recursive_valid_productions = [vp for vp in valid_productions if vp in g.recursive_prods]
        if recursive_valid_productions:
            return recursive_valid_productions
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


def pi_grow_method(
    r: Source,
    g: Grammar,
    depth: int,
    starting_symbol: type[Any] = int,
):
    """PI Grow tree-initialization method.

    (http://ncra.ucd.ie/papers/Exploring%20Position%20Independent%20Initialisation%20in%20Grammatical.%20Evolution.pdf),
    where trees are grown to have at least one branchas deep as possible.
    """
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
        valid_productions = [vp for vp in possible_choices if g.distanceToTerminal[vp] <= depth]
        last_recursive_symbol = nRecs[0] == 0  # Are we the last recursive symbol?
        any_recursive_symbols_in_expansion = any(
            [prod in g.recursive_prods for prod in valid_productions],
        )  # Are there any  recursive symbols in our expansion?

        # If so, then only expand into recursive symbols
        if last_recursive_symbol and any_recursive_symbols_in_expansion:
            valid_productions = [vp for vp in valid_productions if vp in g.recursive_prods]

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


T = TypeVar("T")


def random_list(
    r: Source,
    receiver,
    new_symbol,
    depth: int,
    ty: type[list[T]],
    ctx: dict[str, str],
    prod: str = "",
):
    inner_type = get_generic_parameter(ty)
    size = 1
    if depth > 0:
        size = r.randint(1, depth, prod)
    fins = build_finalizers(lambda *x: receiver(GengyList(inner_type, x)), size)
    ident = ctx["_"]
    for i, fin in enumerate(fins):
        nctx = ctx.copy()
        nident = ident + "_" + str(i)
        nctx["_"] = nident
        new_symbol(inner_type, fin, depth, nident, nctx)


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
    """Creates a random node of a given type (starting_symbol)."""
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
        max_int = sys.maxsize
        min_int = -sys.maxsize
        val = r.normalvariate(0, 1, str(starting_symbol))
        val = round(val)
        val = max(min(val, max_int), min_int)
        SMTResolver.register_const(id, val)
        receiver(val)
        return
    elif starting_symbol is float:
        max_float = sys.float_info.max
        min_float = -sys.float_info.max
        val = r.normalvariate(0, 1, str(starting_symbol))
        valf = max(min(val, max_float), min_float)
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
        random_list(
            r,
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
                            [(vp, g.distanceToTerminal[vp]) for vp in compatible_productions],
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
            li: list[Any] = []
            for argn, _ in args:
                name = id + "_" + argn
                ctx[argn] = name

                def fn(val, name=name):
                    pass

                li.append(fn)

            fins = build_finalizers(
                mk_save_init(starting_symbol, receiver),
                len(args),
                li,
            )
            for i, (argn, argt) in enumerate(args):
                new_symbol(argt, fins[i], depth - 1, id + "_" + argn, ctx)


def mk_save_init(starting_symbol: Any, receiver: Callable):
    """Saves a child as a member of the parent node."""
    if isinstance(starting_symbol, type):
        pass
    elif isinstance(starting_symbol, GengyList):
        starting_symbol = starting_symbol.new_like
    else:
        starting_symbol = type(starting_symbol)

    def fin_recv(*x):
        built = starting_symbol(*x)
        if not is_builtin_class_instance(built):
            built.gengy_init_values = x
        return receiver(built)

    return fin_recv
