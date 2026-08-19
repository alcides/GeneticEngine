from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.solutions.individual import PhenotypicIndividual
from geneticengine.solutions.individual import ConcreteIndividual
from geneticengine.evaluation.parallel import ParallelEvaluator
from geneticengine.evaluation.sequential import SequentialEvaluator
from geneticengine.problems.helpers import is_better
from geneticengine.problems import InvalidFitnessException

from geneticengine.grammar.grammar import extract_grammar
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.representations.tree.treebased import TreeBasedRepresentation


class Root(ABC):
    pass


@dataclass
class Leaf(Root):
    a: int


class TestFitnessHelpers:

    def test_is_better(self):
        g = extract_grammar([Leaf], Root)
        r = NativeRandomSource(0)
        representation = TreeBasedRepresentation(g, MaxDepthDecider(r, g, 2))
        evaluator = SequentialEvaluator()

        a = PhenotypicIndividual(genotype=Leaf(1), representation=representation)
        b = PhenotypicIndividual(genotype=Leaf(2), representation=representation)

        problem = SingleObjectiveProblem(fitness_function=lambda x: x.a, minimize=True)
        [ None for _ in evaluator.evaluate(problem, [a, b])]
        assert is_better(problem, a, b)

        problem = SingleObjectiveProblem(fitness_function=lambda x: x.a, minimize=False)
        [ None for _ in evaluator.evaluate(problem, [a, b])]
        assert not is_better(problem, a, b)

    def test_invalid_fitness(self):
        g = extract_grammar([Leaf], Root)
        r = NativeRandomSource(0)
        representation = TreeBasedRepresentation(g, MaxDepthDecider(r, g, 2))
        evaluator = SequentialEvaluator()

        a = PhenotypicIndividual(genotype=Leaf(2), representation=representation)
        b = PhenotypicIndividual(genotype=Leaf(1), representation=representation)

        def custom_fit(l:Leaf):
            if l.a == 1:
                raise InvalidFitnessException()
            else:
                return a

        problem = SingleObjectiveProblem(fitness_function=custom_fit, minimize=True)
        evaluated = [ ind for ind in evaluator.evaluate(problem, [a, b])]
        assert evaluated == [a]
        assert evaluator.number_of_evaluations() == 2

    def test_invalid_fitness_is_skipped_from_a_stream(self):
        g = extract_grammar([Leaf], Root)
        r = NativeRandomSource(0)
        representation = TreeBasedRepresentation(g, MaxDepthDecider(r, g, 2))
        evaluator = SequentialEvaluator()

        def custom_fit(leaf: Leaf):
            if leaf.a == 1:
                raise InvalidFitnessException()
            return leaf.a

        problem = SingleObjectiveProblem(fitness_function=custom_fit, minimize=True)
        stream = (PhenotypicIndividual(Leaf(value), representation) for value in [1, 2, 1, 3])
        evaluated = list(evaluator.evaluate(problem, stream))
        assert [ind.get_phenotype().a for ind in evaluated] == [2, 3]
        assert evaluator.number_of_evaluations() == 4

    def test_parallel_evaluator_skips_invalid_fitness_from_a_stream(self):
        evaluator = ParallelEvaluator(workers=2)

        def custom_fit(value: int):
            if value == 1:
                raise InvalidFitnessException()
            return value

        problem = SingleObjectiveProblem(fitness_function=custom_fit, minimize=True)
        stream = (ConcreteIndividual(value) for value in [1, 2, 1, 3])
        evaluated = list(evaluator.evaluate(problem, stream))

        assert [ind.get_phenotype() for ind in evaluated] == [2, 3]
        assert evaluator.number_of_evaluations() == 4
