from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
import sys
from typing import Any, Type, TypeVar


from geneticengine.grammar.grammar import Grammar
from geneticengine.random.sources import RandomSource
from geneticengine.solutions.tree import GengyList, TreeNode
from geneticengine.representations.tree.utils import relabel_nodes_of_trees
from geneticengine.grammar.utils import get_arguments, is_builtin_class_instance
from geneticengine.grammar.utils import get_generic_parameter
from geneticengine.grammar.utils import is_generic_list
from geneticengine.exceptions import GeneticEngineError
from geneticengine.grammar.metahandlers.base import MetaHandlerGenerator, is_metahandler


@dataclass
class SynthesisContext:
    depth: int
    nodes: int
    expansions: int


T = TypeVar("T")


class SynthesisDecider(ABC):
    def random_int(self, min_int=-sys.maxsize, max_int=sys.maxsize) -> int: ...
    def random_float(self) -> float: ...  # TODO: range
    def random_str(self) -> str: ...
    def random_bool(self) -> bool: ...
    def random_tuple(self, types) -> tuple: ...
    def random_list(self, type) -> list[Any]: ...
    def choose_alternatives(self, alternatives: list[T], ctx: SynthesisContext) -> T: ...


class BasicSynthesisDecider(SynthesisDecider):
    def __init__(self, random: RandomSource, grammar: Grammar, max_depth=10):
        self.random = random
        self.grammar = grammar
        self.max_depth = 10

    def random_int(self, min_int=-sys.maxsize, max_int=sys.maxsize) -> int:
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

    def choose_alternatives(self, alternatives: list[T], ctx: SynthesisContext) -> T:
        assert len(alternatives) > 0, "No alternatives presented"
        alternatives = [x for x in alternatives if self.grammar.distanceToTerminal[x] <= (self.max_depth - ctx.depth)]
        v = self.random.choice(alternatives)
        sys.stdout.write(f"choice, {len(alternatives)}, {ctx.depth}, {ctx.nodes}, {v}\n")
        return v


def wrap_result(v, grammar):
    relabel_nodes_of_trees(v, grammar)
    if not is_builtin_class_instance(v):
        assert isinstance(v, TreeNode)
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
    random: RandomSource,
    grammar: Grammar,
    starting_symbol: type[Any] = int,
    decider: SynthesisDecider = None,
    dependent_values: dict[str, Any] = None,
    context: SynthesisContext = None,
) -> Any:
    if decider is None:
        decider = BasicSynthesisDecider(random, grammar)  # TODO: remove this
    if dependent_values is None:
        dependent_values = {}
    if context is None:
        assert False

    if starting_symbol is int:
        return decider.random_int()
    elif starting_symbol is float:
        return decider.random_float()
    elif starting_symbol is bool:
        return decider.random_bool()
    elif is_generic_list(starting_symbol):
        inner_type = get_generic_parameter(starting_symbol)
        length = decider.random_int(0, 10)
        nctx = SynthesisContext(context.depth + 1, context.nodes + 1, context.expansions + 1)
        nli = []
        for _ in range(length):
            nv = create_node(random, grammar, inner_type, decider, context=nctx)
            nctx.nodes += number_of_nodes(nv)
            nli.append(nv)
        v: GengyList = GengyList(starting_symbol, nli)
        return wrap_result(v, grammar)
    elif is_metahandler(starting_symbol):
        metahandler: MetaHandlerGenerator = starting_symbol.__metadata__[0]
        base_type = get_generic_parameter(starting_symbol)

        def recurse(typ: type):
            return create_node(
                random=random,
                grammar=grammar,
                starting_symbol=typ,
                decider=decider,
                dependent_values=dependent_values,
                context=context,
            )

        v = metahandler.generate(random, grammar, base_type, recurse, dependent_values)
        return wrap_result(v, grammar)
    else:
        if starting_symbol not in grammar.all_nodes:
            raise GeneticEngineError(
                f"Symbol {starting_symbol} not in grammar rules.",
            )
        elif starting_symbol in grammar.alternatives:
            # Expand abstract type (Non-Terminal)
            compatible_productions = grammar.alternatives[starting_symbol]
            rule = decider.choose_alternatives(compatible_productions, context)
            v = create_node(
                random,
                grammar,
                rule,
                decider,
                context=SynthesisContext(context.depth, context.nodes, context.expansions + 1),
            )
            return wrap_result(v, grammar)
        else:
            # Normal concrete type (Production)
            args = []
            dependent_values = {}
            nctx = SynthesisContext(context.depth + 1, context.nodes + 1, context.expansions + 1)
            for argn, argt in get_arguments(starting_symbol):
                arg = create_node(random, grammar, argt, decider, dependent_values, nctx)
                dependent_values[argn] = arg
                args.append(arg)
                nctx.nodes += number_of_nodes(arg)
            v = apply_constructor(starting_symbol, args)
            return wrap_result(v, grammar)
