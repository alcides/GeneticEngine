from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.solutions.individual import Individual
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
            Individual(genotype=Leaf(1), representation=representation),
            Individual(genotype=Leaf(2), representation=representation),
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

        a = Individual(genotype=Leaf(1), representation=representation)
        b = Individual(genotype=Leaf(2), representation=representation)

        problem = SingleObjectiveProblem(fitness_function=lambda x: x.a, minimize=True)
        evaluator.evaluate(problem, [a, b])
        assert is_better(problem, a, b)

        problem = SingleObjectiveProblem(fitness_function=lambda x: x.a, minimize=False)
        evaluator.evaluate(problem, [a, b])
        assert not is_better(problem, a, b)

    def test_sort(self):
        g = extract_grammar([Leaf], Root)
        r = NativeRandomSource(0)
        representation = TreeBasedRepresentation(g, MaxDepthDecider(r, g, 2))
        evaluator = SequentialEvaluator()

        a = Individual(genotype=Leaf(1), representation=representation)
        b = Individual(genotype=Leaf(3), representation=representation)
        c = Individual(genotype=Leaf(2), representation=representation)
        population = [a, b, c]

        problem = SingleObjectiveProblem(fitness_function=lambda x: x.a, minimize=True)
        evaluator.evaluate(problem, population)
        sorted_population = sort_population(population, problem)
        assert sorted_population[0].get_phenotype().a == 1
        assert sorted_population[1].get_phenotype().a == 2
        assert sorted_population[2].get_phenotype().a == 3
