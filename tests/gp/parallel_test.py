from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from geneticengine.algorithms.callbacks.callback import Callback
from geneticengine.algorithms.callbacks.callback import DebugCallback
from geneticengine.algorithms.gp.gp import GP
from geneticengine.algorithms.gp.operators.stop import GenerationStoppingCriterium
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


class TestCallback(Callback):
    def process_iteration(
        self,
        generation: int,
        population,
        time: float,
        gp: GP,
    ) -> None:
        for ind in population:
            x = ind.genotype
            assert isinstance(x, UnderTest)
            assert isinstance(x.a, Leaf)


class TestParallel:
    def test_parallel(self):
        g = extract_grammar([Leaf, OtherLeaf], UnderTest)
        gp = GP(
            representation=TreeBasedRepresentation(g, 10),
            random_source=NativeRandomSource(seed=123),
            problem=SingleObjectiveProblem(
                fitness_function=lambda x: x.gengy_nodes,
                minimize=True,
            ),
            population_size=20,
            stopping_criterium=GenerationStoppingCriterium(10),
            initializer=FullInitializer(),
            callbacks=[DebugCallback(), TestCallback()],
            evaluator=ParallelEvaluator,
        )
        ind = gp.evolve()
        tree = ind.get_phenotype()
        assert isinstance(tree, UnderTest)
        assert isinstance(tree.a, Leaf)
