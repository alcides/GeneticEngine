from __future__ import annotations
import sys
from typing import Any, Type

from geneticengine.grammar.grammar import Grammar
from geneticengine.random.sources import RandomSource
from geneticengine.solutions.tree import GengyList
from geneticengine.representations.tree.utils import relabel_nodes_of_trees
from geneticengine.grammar.utils import get_arguments
from geneticengine.grammar.utils import get_generic_parameter
from geneticengine.grammar.utils import is_generic_list
from geneticengine.exceptions import GeneticEngineError
from geneticengine.grammar.metahandlers.base import MetaHandlerGenerator, is_metahandler


def relabel(f):
    def g(*args):
        v = f(*args)
        grammar = args[1]
        relabel_nodes_of_trees(v, grammar)
        return v

    return g


def apply_constructor(ty: Type, args: list[Any]):
    v = ty(*args)
    # This saves the metadata used in the constructor for use in mutation and crossover
    if not any(isinstance(v, t) for t in [int, bool, float, str, list]):
        v.gengy_init_values = args
    return v


@relabel
def create_node(
    random: RandomSource,
    grammar: Grammar,
    starting_symbol: type[Any] = int,
    dependent_values: dict[str, Any] = None,
) -> Any:
    if dependent_values is None:
        dependent_values = {}
    if starting_symbol is int:
        max_int = sys.maxsize
        min_int = -sys.maxsize
        val = random.normalvariate(0, 100, str(starting_symbol))
        val = round(val)
        val = max(min(val, max_int), min_int)
        return val
    elif starting_symbol is float:
        max_float = sys.float_info.max
        min_float = -sys.float_info.max
        val = random.normalvariate(0, 100, str(starting_symbol))
        valf = max(min(val, max_float), min_float)
        return valf
    elif starting_symbol is bool:
        valb = random.random_bool(str(starting_symbol))
        return valb
    elif is_generic_list(starting_symbol):
        inner_type = get_generic_parameter(starting_symbol)
        size = random.randint(1, 10)
        li = []
        for _ in range(size):
            child = create_node(random, grammar, inner_type)
            li.append(child)
        return GengyList(inner_type, li)
    elif is_metahandler(starting_symbol):
        metahandler: MetaHandlerGenerator = starting_symbol.__metadata__[0]
        base_type = get_generic_parameter(starting_symbol)
        return metahandler.generate(random, grammar, base_type, create_node, dependent_values)
    else:
        print(starting_symbol, repr(starting_symbol), is_metahandler(starting_symbol))
        if starting_symbol not in grammar.all_nodes:
            raise GeneticEngineError(
                f"Symbol {starting_symbol} not in grammar rules.",
            )
        elif starting_symbol in grammar.alternatives:
            compatible_productions = grammar.alternatives[starting_symbol]
            real_options = [
                x for x in compatible_productions if grammar.distanceToTerminal[x] <= 1
            ]  # TODO TreeManager: Add tree manager
            rule = random.choice(real_options or compatible_productions)
            return create_node(random, grammar, rule)
        else:  # Normal production
            args = []
            dependent_values = {}
            for argn, argt in get_arguments(starting_symbol):
                arg = create_node(random, grammar, argt, dependent_values)
                dependent_values[argn] = arg
                args.append(arg)
            return apply_constructor(starting_symbol, args)
