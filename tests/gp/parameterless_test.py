from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from geneticengine.algorithms.gp.adaptive import AdaptiveGeneticProgramming
from geneticengine.algorithms.gp.parameterless import AlwaysRandomGeneticProgramming, InitiallyRandomGeneticProgramming
from geneticengine.evaluation.budget import TimeBudget

from geneticengine.grammar.grammar import extract_grammar
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import ProgressivelyTerminalDecider
from geneticengine.representations.tree.treebased import TreeBasedRepresentation


class Root(ABC):
    pass


@dataclass(unsafe_hash=True)
class Option(Root):
    a: int


def fitness_function(r: Root) -> float:
    assert isinstance(r, Option)
    return r.a


class TestParameterless:
    def test_adaptive(self):

        grammar = extract_grammar([Option], Root)
        random = NativeRandomSource(1)
        decider = ProgressivelyTerminalDecider(random, grammar)
        gp = AdaptiveGeneticProgramming(
            random=random,
            representation=TreeBasedRepresentation(grammar, decider=decider),
            budget=TimeBudget(4),
            problem=SingleObjectiveProblem(fitness_function=fitness_function),
        )
        gp.search()

        assert gp.tracker.evaluator.number_of_evaluations() >= 0

    def test_intially_random(self):

        grammar = extract_grammar([Option], Root)
        random = NativeRandomSource(1)
        decider = ProgressivelyTerminalDecider(random, grammar)
        gp = InitiallyRandomGeneticProgramming(
            random=random,
            representation=TreeBasedRepresentation(grammar, decider=decider),
            budget=TimeBudget(4),
            problem=SingleObjectiveProblem(fitness_function=fitness_function),
        )
        gp.search()

        assert gp.tracker.evaluator.number_of_evaluations() >= 0

    def test_always_random(self):

        grammar = extract_grammar([Option], Root)
        random = NativeRandomSource(1)
        decider = ProgressivelyTerminalDecider(random, grammar)
        gp = AlwaysRandomGeneticProgramming(
            random=random,
            representation=TreeBasedRepresentation(grammar, decider=decider),
            budget=TimeBudget(4),
            problem=SingleObjectiveProblem(fitness_function=fitness_function),
        )
        gp.search()

        assert gp.tracker.evaluator.number_of_evaluations() >= 0
