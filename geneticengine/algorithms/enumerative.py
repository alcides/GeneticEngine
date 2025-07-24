from __future__ import annotations
from dataclasses import is_dataclass
from itertools import count, takewhile
from typing import Any, Generator, Optional

from geneticengine.algorithms.api import SynthesisAlgorithm

from geneticengine.evaluation.budget import SearchBudget
from geneticengine.evaluation.tracker import ProgressTracker
from geneticengine.exceptions import GeneticEngineError
from geneticengine.grammar.grammar import Grammar
from geneticengine.grammar.metahandlers.base import MetaHandlerGenerator
from geneticengine.grammar.utils import (
    get_arguments,
    get_generic_parameter,
    get_generic_parameters,
    is_generic_list,
    is_generic_tuple,
    is_metahandler,
    is_union,
)
from geneticengine.problems import Problem
from geneticengine.representations.tree.initializations import apply_constructor
from geneticengine.solutions.individual import (
    ConcreteIndividual,
    Individual,
)

def frange(start, stop, step):
    return takewhile(lambda x: x < stop, count(start, step))

def foundDict(d, target, result):
    for key,value in d.items():
        if target == value[0]:
            d[key][1] = result
            return


# This needs to be redone to use a proper order
def combine_list_types(ts: list[type], acc: list[Any], gen, dependent_values: Optional[dict[str,Any]]={}):
    match ts:
        case []:
            yield acc
        case _:
            head = ts[0]
            tail = ts[1:]
            for x in gen(head,dependent_values):
                foundDict(dependent_values, head, x)
                yield from combine_list_types(tail, acc + [x], gen, dependent_values)


def iterate_grammar(grammar: Grammar, starting_symbol: type, level:int, generator_for_recursive: Optional[Any] = None, dependent_values: Optional[dict[str,Any]] = {}):
    if level < 0:
        return
    def generator_for_recursive(symbol: type, dependent_values: Optional[dict[str,Any]] = {}):
        print(level, level)
        yield from iterate_grammar(grammar, symbol, level-1, generator_for_recursive, dependent_values)

    if starting_symbol is int and level == 0:
        yield from range(-100000000, 100000000)
    elif starting_symbol is float and level == 0:
        yield from frange(-100000.0, 100000.0, 0.00001)
    elif starting_symbol is bool and level == 0:
        yield True
        yield False
    elif is_generic_tuple(starting_symbol):
        types = get_generic_parameters(starting_symbol)
        for li in combine_list_types(types, [], generator_for_recursive, {}):
            yield tuple(li)
    elif is_generic_list(starting_symbol):
        inner_type = get_generic_parameter(starting_symbol)

        length = 0
        for _ in range(1000):
            generator_list = [inner_type for _ in range(length)]
            for concrete_list in combine_list_types(generator_list, [], generator_for_recursive, {}):
                yield concrete_list
            length += 1

    elif is_metahandler(starting_symbol):
        metahandler: MetaHandlerGenerator = starting_symbol.__metadata__[0]  # type: ignore
        base_type = get_generic_parameter(starting_symbol)

        if hasattr(metahandler, "iterate"):
            yield from metahandler.iterate(base_type, lambda xs: combine_list_types(xs, [], generator_for_recursive, dependent_values), generator_for_recursive, dependent_values)
        else:
            base_type = get_generic_parameter(starting_symbol)
            for ins in generator_for_recursive(base_type):
                if metahandler.validate(ins):
                    yield ins
    elif is_union(starting_symbol):
        for alt in get_generic_parameters(starting_symbol):
            yield from generator_for_recursive(alt)
    else:
        if starting_symbol not in grammar.all_nodes:
            raise GeneticEngineError(
                f"Symbol {starting_symbol} not in grammar rules.",
            )
        elif starting_symbol in grammar.alternatives:
            compatible_productions = [ s
                                      for s in grammar.alternatives[starting_symbol]
                                      if grammar.distanceToTerminal[s] <= level
            ]
            for prod in compatible_productions:
                yield from generator_for_recursive(prod, dependent_values)

        elif is_dataclass(starting_symbol):
            # Normal production
            args = []
            dependent_values = {}
            if get_arguments(starting_symbol):
                for _, argt in get_arguments(starting_symbol):
                    args.append(argt)
                for li in combine_list_types(args, [], generator_for_recursive, dependent_values):
                    yield apply_constructor(starting_symbol, li)
            elif level == 0:
                yield apply_constructor(starting_symbol, [])



def iterate_individuals(grammar: Grammar, starting_symbol: type) -> Generator[ConcreteIndividual, Any, Any]:
    for level in range(100):
        for p in iterate_grammar(grammar, starting_symbol, level):
            yield ConcreteIndividual(instance=p)


class EnumerativeSearch(SynthesisAlgorithm):
    """Iterates through all possible representations and selects the best."""

    def __init__(
        self,
        problem: Problem,
        budget: SearchBudget,
        grammar: Grammar,
        tracker: ProgressTracker | None = None,
    ):
        super().__init__(problem, budget, tracker)
        self.grammar = grammar

    def perform_search(self) -> list[Individual]:
        for individual in iterate_individuals(self.grammar, self.grammar.starting_symbol):
            self.tracker.evaluate_single(individual)
            if self.is_done():
                break
        return self.tracker.get_best_individuals()
