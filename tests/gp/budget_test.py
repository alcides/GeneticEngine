from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.evaluation.budget import EvaluationBudget

from geneticengine.grammar.grammar import extract_grammar
from geneticengine.problems import LazyMultiObjectiveProblem, SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.representations.tree.treebased import TreeBasedRepresentation


class Root(ABC):
    pass


@dataclass(unsafe_hash=True)
class Option(Root):
    a: int


def fitness_function(r: Root) -> float:
    assert isinstance(r, Option)
    return r.a


def fitness_function_multi(r: Root) -> list[float]:
    assert isinstance(r, Option)
    return [r.a]


class TestBudget:
    def test_evaluations(self):
        limit = 60
        population_size = 11

        grammar = extract_grammar([Option], Root)
        random = NativeRandomSource(1)
        decider = MaxDepthDecider(random, grammar, max_depth=2)
        gp = GeneticProgramming(
            representation=TreeBasedRepresentation(grammar, decider=decider),
            budget=EvaluationBudget(limit),
            problem=SingleObjectiveProblem(fitness_function=fitness_function),
            population_size=population_size,
        )
        gp.search()

        assert gp.tracker.evaluator.number_of_evaluations() <= limit + population_size

    def test_evaluationsmultiobjective(self):
        limit = 33
        population_size = 11

        grammar = extract_grammar([Option], Root)
        random = NativeRandomSource(1)
        decider = MaxDepthDecider(random, grammar, max_depth=2)
        gp = GeneticProgramming(
            representation=TreeBasedRepresentation(grammar, decider=decider),
            budget=EvaluationBudget(limit),
            problem=LazyMultiObjectiveProblem(minimize=False, fitness_function=fitness_function_multi),
            population_size=population_size,
        )
        gp.search()

        assert gp.tracker.evaluator.number_of_evaluations() < limit + population_size
