from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.algorithms.gp.operators.combinators import SequenceStep
from geneticengine.algorithms.gp.operators.crossover import GenericCrossoverStep
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.algorithms.gp.operators.selection import TournamentSelection
from geneticengine.evaluation.budget import EvaluationBudget
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.representations.tree.operators import FullInitializer
from geneticengine.representations.tree.treebased import TreeBasedRepresentation


class Root(ABC):
    pass


@dataclass
class Leaf(Root):
    pass


@dataclass
class OtherLeaf(Root):
    pass


@dataclass
class UnderTest(Root):
    a: Leaf
    b: Root


def fitness_function(x):
    assert isinstance(x, UnderTest)
    assert isinstance(x.a, Leaf)
    return x.gengy_nodes


class TestGrammar:
    def test_safety(self):
        max_depth = 10
        r = NativeRandomSource(seed=123)
        g = extract_grammar([Leaf, OtherLeaf], UnderTest)
        gp = GeneticProgramming(
            problem=SingleObjectiveProblem(
                fitness_function=lambda x: x.gengy_nodes,
                minimize=True,
            ),
            budget=EvaluationBudget(100),
            representation=TreeBasedRepresentation(g, MaxDepthDecider(r, g, max_depth)),
            random=r,
            population_size=20,
            population_initializer=FullInitializer(max_depth=max_depth),
            step=SequenceStep(
                TournamentSelection(10),
                GenericCrossoverStep(1),
                GenericMutationStep(1),
            ),
        )
        ind = gp.search()[0]
        tree = ind.get_phenotype()
        assert isinstance(tree, UnderTest)
        assert isinstance(tree.a, Leaf)
        assert "generation" in ind.metadata
