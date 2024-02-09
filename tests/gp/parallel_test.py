from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.evaluation.budget import EvaluationBudget
from geneticengine.evaluation.recorder import SingleObjectiveProgressTracker
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.operators import FullInitializer
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.evaluation.parallel import ParallelEvaluator


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


# TODO
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


class TestParallel:
    def test_parallel(self):
        g = extract_grammar([Leaf, OtherLeaf], UnderTest)
        p = SingleObjectiveProblem(
            fitness_function=lambda x: 3,
            minimize=True,
        )
        gp = GeneticProgramming(
            representation=TreeBasedRepresentation(g, 10),
            random=NativeRandomSource(seed=123),
            problem=p,
            population_size=20,
            budget=EvaluationBudget(100),
            population_initializer=FullInitializer(),
            recorder=SingleObjectiveProgressTracker(problem=p, evaluator=ParallelEvaluator()),
        )
        ind = gp.search()
        tree = ind.get_phenotype()
        assert isinstance(tree, UnderTest)
        assert isinstance(tree.a, Leaf)
