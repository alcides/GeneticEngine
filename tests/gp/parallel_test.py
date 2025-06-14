from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any

from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.evaluation.budget import EvaluationBudget
from geneticengine.evaluation.recorder import SearchRecorder
from geneticengine.evaluation.tracker import ProgressTracker
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.problems import Problem, SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.representations.tree.operators import FullInitializer
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.evaluation.parallel import ParallelEvaluator
from geneticengine.solutions.individual import Individual, PhenotypicIndividual


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


class TestRecorder(SearchRecorder):
    def register(self, tracker: Any, individual: Individual, problem: Problem, is_best: bool):
        assert isinstance(individual, PhenotypicIndividual)
        x = individual.genotype
        assert isinstance(x, UnderTest)
        assert isinstance(x.a, Leaf)


class TestParallel:
    def test_parallel(self):
        max_depth = 10
        g = extract_grammar([Leaf, OtherLeaf], UnderTest)
        p = SingleObjectiveProblem(
            fitness_function=lambda x: 3,
            minimize=True,
        )
        r = NativeRandomSource(seed=123)
        tracker = ProgressTracker(problem=p, evaluator=ParallelEvaluator(), recorders=[TestRecorder()])
        gp = GeneticProgramming(
            representation=TreeBasedRepresentation(
                g,
                MaxDepthDecider(r, g, max_depth=max_depth),
            ),
            random=r,
            problem=p,
            population_size=20,
            budget=EvaluationBudget(100),
            population_initializer=FullInitializer(max_depth=max_depth),
            tracker=tracker,
        )
        ind = gp.search()[0]
        tree = ind.get_phenotype()
        assert isinstance(tree, UnderTest)
        assert isinstance(tree.a, Leaf)
