from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.solutions.individual import ConcreteIndividual, PhenotypicIndividual
from geneticengine.evaluation.sequential import SequentialEvaluator
from geneticengine.problems.helpers import best_individual, is_better, sort_population

from geneticengine.grammar.grammar import extract_grammar
from geneticengine.problems import MultiObjectiveProblem, SingleObjectiveProblem
from geneticengine.representations.tree.treebased import TreeBasedRepresentation


class Root(ABC):
    pass


@dataclass
class Leaf(Root):
    a: int


class TestFitnessHelpers:
    def test_best_individual(self):
        g = extract_grammar([Leaf], Root)
        r = NativeRandomSource(0)
        representation = TreeBasedRepresentation(g, MaxDepthDecider(r, g, 2))
        evaluator = SequentialEvaluator()

        population = [
            PhenotypicIndividual(genotype=Leaf(1), representation=representation),
            PhenotypicIndividual(genotype=Leaf(2), representation=representation),
        ]

        problem = SingleObjectiveProblem(fitness_function=lambda x: x.a, minimize=False)
        evaluator.evaluate(problem, population)
        x = best_individual(population, problem)
        assert x.get_phenotype().a == 2

        problem = SingleObjectiveProblem(fitness_function=lambda x: x.a, minimize=True)
        evaluator.evaluate(problem, population)
        x = best_individual(population, problem)
        assert x.get_phenotype().a == 1

        problem = MultiObjectiveProblem(minimize=[True, True], fitness_function=lambda x: [x.a, x.a])
        evaluator.evaluate(problem, population)
        x = best_individual(population, problem)
        assert x.get_phenotype().a == 1

    def test_is_better(self):
        g = extract_grammar([Leaf], Root)
        r = NativeRandomSource(0)
        representation = TreeBasedRepresentation(g, MaxDepthDecider(r, g, 2))
        evaluator = SequentialEvaluator()

        a = PhenotypicIndividual(genotype=Leaf(1), representation=representation)
        b = PhenotypicIndividual(genotype=Leaf(2), representation=representation)

        problem = SingleObjectiveProblem(fitness_function=lambda x: x.a, minimize=True)
        evaluator.evaluate(problem, [a, b])
        assert is_better(problem, a, b)

        problem = SingleObjectiveProblem(fitness_function=lambda x: x.a, minimize=False)
        evaluator.evaluate(problem, [a, b])
        assert not is_better(problem, a, b)

    def test_sort(self):
        evaluator = SequentialEvaluator()

        a = ConcreteIndividual(instance=Leaf(1))
        b = ConcreteIndividual(instance=Leaf(3))
        c = ConcreteIndividual(instance=Leaf(2))
        population = [a, b, c]

        problem = SingleObjectiveProblem(fitness_function=lambda x: x.a, minimize=True)
        evaluator.evaluate(problem, population)
        sorted_population = sort_population(population, problem)
        assert sorted_population[0].get_phenotype().a == 1
        assert sorted_population[1].get_phenotype().a == 2
        assert sorted_population[2].get_phenotype().a == 3



def test_hyper_vol():
    p = MultiObjectiveProblem(
        fitness_function=lambda _: [0, 1, 10],
        minimize=[False, False, False],
    )

    ind = ConcreteIndividual(instance=None)

    evaluator = SequentialEvaluator()
    evaluator.evaluate(p, [ ind])

    assert ind.get_fitness().maximizing_aggregate > 0


def test_hyper_vol_min():
    p = MultiObjectiveProblem(
        fitness_function=lambda _: [0, 1, 10],
        minimize=[True, True, True],
    )

    ind = ConcreteIndividual(instance=None)

    evaluator = SequentialEvaluator()
    evaluator.evaluate(p, [ ind])

    assert ind.get_fitness().maximizing_aggregate < 0
