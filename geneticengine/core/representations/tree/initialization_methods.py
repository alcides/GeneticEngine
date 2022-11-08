from __future__ import annotations

from typing import Any
from typing import Callable
from abc import ABC, abstractmethod
import z3
from geneticengine.core.decorators import get_gengy

from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.tree.utils import GengyList, relabel_nodes_of_trees
from geneticengine.core.utils import build_finalizers, get_arguments, get_generic_parameter, is_abstract, is_generic_list
from geneticengine.exceptions import GeneticEngineError
from geneticengine.metahandlers.base import is_metahandler


class Initialization_Method(ABC):
    min_depth: int | None = None
    
    @abstractmethod
    def tree_init_method(self, r: Source, g: Grammar, max_depth: int, starting_symbol: type[Any]):
        ...

class Random_Production(Initialization_Method):
    def __init__(
        self, 
        min_depth: int | None = None
    ):
        self.min_depth = min_depth
    
    def tree_init_method(
        self,
        r: Source,
        g: Grammar,
        max_depth: int,
        starting_symbol: type[Any] = int,
    ):
        if self.min_depth:
            assert self.min_depth <= max_depth
            def filter_choices(possible_choices: list[type], depth):
                current_depth = max_depth - depth
                valid_productions = [
                    vp for vp in possible_choices if g.distanceToTerminal[vp] <= depth
                ]
                if (nRecs[0] == 0) and any([  # Are we the last recursive symbol?
                        prod in g.recursive_prods for prod in valid_productions
                    ]  # Are there any  recursive symbols in our expansion?
                ) and current_depth <= self.min_depth:
                    valid_productions = [
                        vp for vp in valid_productions if vp in g.recursive_prods
                    ]  # If so, then only expand into recursive symbols

                return valid_productions
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

            handle_symbol(starting_symbol, final_finalize, max_depth, "root", ctx={})
            
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
        else:
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

            handle_symbol(starting_symbol, final_finalize, max_depth, "root", ctx={})
            SMTResolver.resolve_clauses()
            n = state["final"]
            relabel_nodes_of_trees(n, g)
            return n

    

class PI_Grow(Initialization_Method):
    def tree_init_method(
        self,
        r: Source,
        g: Grammar,
        max_depth: int,
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

        handle_symbol(starting_symbol, final_finalize, max_depth, "root", ctx={})

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

