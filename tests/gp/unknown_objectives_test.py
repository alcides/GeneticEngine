from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Annotated

from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.algorithms.gp.operators.combinators import SequenceStep
from geneticengine.algorithms.gp.operators.crossover import GenericCrossoverStep
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.algorithms.gp.operators.selection import LexicaseSelection, TournamentSelection
from geneticengine.evaluation.budget import AnyOf, EvaluationBudget, TargetMultiFitness
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.problems import MultiObjectiveProblem, SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.operators import FullInitializer
from geneticengine.representations.tree.treebased import TreeBasedRepresentation


class Root(ABC):
    pass


@dataclass
class Leaf(Root):
    a:Annotated[int, IntRange(0, 10)]
    b:Annotated[int, IntRange(0, 10)]

def fitness_function(x):
    return [x.a, x.b]


class TestUnknownObjectives:
    def test_unknown_objectives_and_target(self):
        g = extract_grammar([Leaf], Root)
        gp = GeneticProgramming(
            problem=MultiObjectiveProblem(
                fitness_function=fitness_function,
                minimize=[],
            ),
            budget=AnyOf(EvaluationBudget(100), TargetMultiFitness([])),
            representation=TreeBasedRepresentation(g, 10),
            random=NativeRandomSource(seed=123),
            population_size=20,
            population_initializer=FullInitializer(),
            step=SequenceStep(
                LexicaseSelection(10),
                GenericCrossoverStep(1),
                GenericMutationStep(1),
            ),
        )
        ind = gp.search()
        tree = ind.get_phenotype()
        assert isinstance(tree, Root)
        assert isinstance(tree.a, int)
        assert tree.a > 0 and tree.b > 0
