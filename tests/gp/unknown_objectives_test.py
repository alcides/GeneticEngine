from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Annotated

from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.algorithms.gp.operators.combinators import SequenceStep
from geneticengine.algorithms.gp.operators.crossover import GenericCrossoverStep
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.algorithms.gp.operators.selection import LexicaseSelection
from geneticengine.evaluation.budget import EvaluationBudget
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.problems import MultiObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.representations.tree.operators import FullInitializer
from geneticengine.representations.tree.treebased import TreeBasedRepresentation


class Root(ABC):
    pass


@dataclass
class Leaf(Root):
    a: Annotated[int, IntRange(0, 10)]
    b: Annotated[int, IntRange(0, 10)]


def fitness_function(x):
    return [x.a, x.b]


class TestUnknownObjectives:
    def test_unknown_objectives_and_target(self):
        g = extract_grammar([Leaf], Root)
        max_depth = 10
        r = NativeRandomSource(seed=123)
        gp = GeneticProgramming(
            problem=MultiObjectiveProblem(fitness_function=fitness_function, minimize=[], target=[4, 5]),
            budget=EvaluationBudget(100),
            representation=TreeBasedRepresentation(g, MaxDepthDecider(r, g, max_depth)),
            random=r,
            population_size=20,
            population_initializer=FullInitializer(max_depth=max_depth),
            step=SequenceStep(
                LexicaseSelection(False),
                GenericCrossoverStep(1),
                GenericMutationStep(1),
            ),
        )
        ind = gp.search()[0]
        tree = ind.get_phenotype()
        assert isinstance(tree, Root)
        assert isinstance(tree, Leaf)
        assert isinstance(tree.a, int)
        assert 0 <= tree.a < 10 and 0 <= tree.b < 10

    def test_unknown_objectives_and_unkown_target(self):
        max_depth = 10
        g = extract_grammar([Leaf], Root)
        r = NativeRandomSource(seed=123)
        gp = GeneticProgramming(
            problem=MultiObjectiveProblem(fitness_function=fitness_function, minimize=[], target=[10]),
            budget=EvaluationBudget(100),
            representation=TreeBasedRepresentation(g, MaxDepthDecider(r, g, max_depth=max_depth)),
            random=r,
            population_size=20,
            population_initializer=FullInitializer(max_depth=max_depth),
            step=SequenceStep(
                LexicaseSelection(False),
                GenericCrossoverStep(1),
                GenericMutationStep(1),
            ),
        )
        ind = gp.search()[0]
        tree = ind.get_phenotype()
        assert isinstance(tree, Root)
        assert isinstance(tree, Leaf)
        assert isinstance(tree.a, int)
        assert 0 <= tree.a < 10 and 0 <= tree.b < 10
