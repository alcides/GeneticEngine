from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
from math import log10
import sys
from typing import Any, Type, TypeVar


from geneticengine.grammar.grammar import Grammar
from geneticengine.random.sources import RandomSource
from geneticengine.solutions.tree import GengyList, LocalSynthesisContext, TreeNode
from geneticengine.representations.tree.utils import relabel_nodes_of_trees
from geneticengine.grammar.utils import get_arguments, is_builtin_class_instance
from geneticengine.grammar.utils import is_union, get_generic_parameters
from geneticengine.grammar.utils import get_generic_parameter
from geneticengine.grammar.utils import is_generic_list
from geneticengine.exceptions import GeneticEngineError
from geneticengine.grammar.metahandlers.base import MetaHandlerGenerator, SynthesisException, is_metahandler


@dataclass
class GlobalSynthesisContext:
    random: RandomSource
    grammar: Grammar
    decider: SynthesisDecider


T = TypeVar("T")


class SynthesisDecider(ABC):
    def random_int(self, min_int=-sys.maxsize, max_int=sys.maxsize) -> int: ...
    def random_float(self) -> float: ...
    def random_str(self) -> str: ...
    def random_bool(self) -> bool: ...
    def random_tuple(self, types) -> tuple: ...
    def random_list(self, type) -> list[Any]: ...
    def choose_production_alternatives(self, alternatives: list[type], ctx: LocalSynthesisContext) -> type: ...
    def choose_options(self, alternatives: list[T], ctx: LocalSynthesisContext) -> T: ...


class BasicSynthesisDecider(SynthesisDecider):
    def __init__(self, random: RandomSource, grammar: Grammar, max_depth=10):
        self.random = random
        self.grammar = grammar
        self.max_depth = 10

    def random_int(self, min_int=-sys.maxsize, max_int=sys.maxsize) -> int:
        width = max_int - min_int
        if width > 1000:
            n = self.random.randint(0, 10)
            e = self.random.randint(0, round(log10(width)))
            extra = (n ^ e) % width
            v = min_int + extra
            return v
        return self.random.randint(min_int, max_int)

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

    def choose_production_alternatives(self, alternatives: list[type], ctx: LocalSynthesisContext) -> type:
        assert len(alternatives) > 0, "No alternatives presented"
        alternatives = [
            x for x in alternatives if self.grammar.get_distance_to_terminal(x) <= (self.max_depth - ctx.depth)
        ]
        return self.random.choice(alternatives)

    def choose_options(self, alternatives: list[T], ctx: LocalSynthesisContext) -> T:
        assert len(alternatives) > 0, "No alternatives presented"
        return self.random.choice(alternatives)


def wrap_result(
    v: Any,
    global_context: GlobalSynthesisContext,
    context: LocalSynthesisContext,
) -> TreeNode:
    if not is_builtin_class_instance(v):
        relabel_nodes_of_trees(v, global_context.grammar)
        v.gengy_synthesis_context = context
    return v


def apply_constructor(ty: Type, args: list[Any]):
    v = ty(*args)
    # This saves the metadata used in the constructor for use in mutation and crossover
    if not any(isinstance(v, t) for t in [int, bool, float, str, list]):
        v.gengy_init_values = args
    return v


def number_of_nodes(v: TreeNode):
    if hasattr(v, "gengy_nodes"):
        return v.gengy_nodes
    else:
        return 1


def create_node(
    global_context: GlobalSynthesisContext,
    starting_symbol: type[Any],
    context: LocalSynthesisContext,
    dependent_values: dict[str, Any] | None = None,
    initial_values: dict[str, TreeNode] | None = None,
) -> Any:
    dependent_vals: dict[str, Any] = dependent_values if dependent_values is not None else {}
    initial_vals: dict[str, TreeNode] = initial_values if initial_values is not None else {}

    decider = global_context.decider

    if starting_symbol is int:
        return decider.random_int()
    elif starting_symbol is float:
        return decider.random_float()
    elif starting_symbol is bool:
        return decider.random_bool()
    elif is_generic_list(starting_symbol):
        inner_type = get_generic_parameter(starting_symbol)
        length = decider.random_int(0, 10)
        nctx = LocalSynthesisContext(context.depth + 1, context.nodes + 1, context.expansions + 1, dependent_vals)
        nli = []
        for _ in range(length):
            nv = create_node(global_context, inner_type, nctx)
            nctx.nodes += number_of_nodes(nv)
            nli.append(nv)
        vl: GengyList = GengyList(starting_symbol, nli)
        return wrap_result(vl, global_context, context)
    elif is_metahandler(starting_symbol):
        metahandler: MetaHandlerGenerator = starting_symbol.__metadata__[0]
        base_type = get_generic_parameter(starting_symbol)

        def recurse(typ: type, **kwargs):
            v = create_node(
                global_context,
                starting_symbol=typ,
                dependent_values=dependent_values,
                context=LocalSynthesisContext(context.depth, context.nodes, context.expansions + 1, dependent_vals),
                **kwargs,
            )
            return v

        v = metahandler.generate(global_context.random, global_context.grammar, base_type, recurse, dependent_vals)
        return wrap_result(v, global_context, context)
    elif is_union(starting_symbol):
        t: type = decider.choose_production_alternatives(get_generic_parameters(starting_symbol), context)
        v = create_node(global_context, t, context, dependent_values, initial_values)
        return wrap_result(v, global_context, context)
    else:
        if starting_symbol not in global_context.grammar.all_nodes:
            raise GeneticEngineError(
                f"Symbol {starting_symbol} not in grammar rules.",
            )
        elif starting_symbol in global_context.grammar.alternatives:
            # Expand abstract type (Non-Terminal)
            compatible_productions = global_context.grammar.alternatives[starting_symbol]
            while compatible_productions:
                rule = decider.choose_production_alternatives(compatible_productions, context)
                try:
                    v = create_node(
                        global_context,
                        rule,
                        context=LocalSynthesisContext(
                            context.depth,
                            context.nodes,
                            context.expansions + 1,
                            dependent_vals,
                        ),
                        initial_values=initial_vals,
                    )
                    return wrap_result(v, global_context, context)
                except SynthesisException:
                    compatible_productions.remove(rule)
            raise SynthesisException(f"Could not find any suitable alternative for {starting_symbol}")
        else:
            # Normal concrete type (Production)
            args = []
            dependent_values = {}
            nctx = LocalSynthesisContext(context.depth + 1, context.nodes + 1, context.expansions + 1, dependent_vals)
            for argn, argt in get_arguments(starting_symbol):
                arg: TreeNode
                if argn in initial_vals:
                    arg = initial_vals.get(argn)  # pyright: ignore
                    if isinstance(arg, list):
                        list_type = argt
                        inner_type = get_generic_parameter(list_type)
                        arg = GengyList(inner_type, arg)  # type: ignore
                else:
                    arg = create_node(
                        global_context,
                        argt,
                        nctx,
                        dependent_values,
                    )
                dependent_values[argn] = arg
                args.append(arg)
                nctx.nodes += number_of_nodes(arg)
            v = apply_constructor(starting_symbol, args)
            return wrap_result(v, global_context, context)
