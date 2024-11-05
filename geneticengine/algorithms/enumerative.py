from __future__ import annotations
from itertools import count, takewhile
from typing import Any

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


def combine_list_types(ts: list[type], acc: list[Any], gen):
    match ts:
        case []:
            yield acc
        case _:
            head = ts[0]
            tail = ts[1:]
            for x in gen(head):
                yield from combine_list_types(tail, acc + [x], gen)


def iterate_grammar(grammar: Grammar, starting_symbol: type):
    def rec_generator(symbol):
        return iterate_grammar(grammar, symbol)

    if starting_symbol is int:
        yield from range(-100000000, 100000000)
    elif starting_symbol is float:
        yield from frange(-100000.0, 100000.0, 0.00001)
    elif starting_symbol is bool:
        yield True
        yield False
    elif is_generic_tuple(starting_symbol):
        types = get_generic_parameters(starting_symbol)
        for li in combine_list_types(types, [], rec_generator):
            yield tuple(li)
    elif is_generic_list(starting_symbol):
        inner_type = get_generic_parameter(starting_symbol)

        for length in range(0, 1024):

            generator_list = [inner_type for _ in range(length)]
            for concrete_list in combine_list_types(generator_list, [], rec_generator):
                yield concrete_list

    elif is_metahandler(starting_symbol):
        metahandler: MetaHandlerGenerator = starting_symbol.__metadata__[0]  # type: ignore
        base_type = get_generic_parameter(starting_symbol)

        if hasattr(metahandler, "iterate"):
            yield from metahandler.iterate(base_type, lambda xs: combine_list_types(xs, [], rec_generator))
        else:
            base_type = get_generic_parameter(starting_symbol)
            for ins in iterate_grammar(grammar, base_type):
                if metahandler.validate(ins):
                    yield ins
    elif is_union(starting_symbol):
        for alt in get_generic_parameters(starting_symbol):
            yield from iterate_grammar(grammar, alt)
    else:
        if starting_symbol not in grammar.all_nodes:
            raise GeneticEngineError(
                f"Symbol {starting_symbol} not in grammar rules.",
            )
        elif starting_symbol in grammar.alternatives:
            compatible_productions = grammar.alternatives[starting_symbol]

            for prod in sorted(compatible_productions, key=lambda x: grammar.distanceToTerminal[x]):
                yield from iterate_grammar(grammar, prod)
        else:
            # Normal production
            args = []
            # TODO: Add dependent types to enumerative
            # dependent_values = {}
            args = [argt for _, argt in get_arguments(starting_symbol)]
            for li in combine_list_types(args, [], rec_generator):
                yield apply_constructor(starting_symbol, li)


def iterate_individuals(grammar: Grammar, starting_symbol: type):
    for p in iterate_grammar(grammar, starting_symbol):
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
