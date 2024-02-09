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


# TODO: Callback
class TestCallback:
    def process_iteration(
        self,
        generation: int,
        population,
        time: float,
        gp: GeneticProgramming,
    ) -> None:
        for ind in population:
            x = ind.genotype
            assert isinstance(x, UnderTest)
            assert isinstance(x.a, Leaf)


class TestGrammar:
    def test_safety(self):
        g = extract_grammar([Leaf, OtherLeaf], UnderTest)
        gp = GeneticProgramming(
            problem=SingleObjectiveProblem(
                fitness_function=lambda x: x.gengy_nodes,
                minimize=True,
            ),
            budget=EvaluationBudget(100),
            representation=TreeBasedRepresentation(g, 10),
            random=NativeRandomSource(seed=123),
            population_size=20,
            population_initializer=FullInitializer(),
            step=SequenceStep(
                TournamentSelection(10),
                GenericCrossoverStep(1),
                GenericMutationStep(1),
            ),
        )
        ind = gp.search()
        tree = ind.get_phenotype()
        assert isinstance(tree, UnderTest)
        assert isinstance(tree.a, Leaf)
