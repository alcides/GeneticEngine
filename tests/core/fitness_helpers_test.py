from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.solutions.individual import PhenotypicIndividual
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
        assert problem.is_better(evaluated[0].get_fitness(problem), evaluated[1].get_fitness(problem))
