from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from geneticengine.algorithms.gp.individual import Individual
from geneticengine.core.evaluators import SequentialEvaluator
from geneticengine.core.fitness_helpers import best_individual, is_better, sort_population

from geneticengine.core.grammar import extract_grammar
from geneticengine.core.problems import MultiObjectiveProblem, SingleObjectiveProblem
from geneticengine.core.representations.tree.treebased import TreeBasedRepresentation


class Root(ABC):
    pass


@dataclass
class Leaf(Root):
    a: int


class TestFitnessHelpers:
    def test_best_individual(self):
        g = extract_grammar([Leaf], Root)
        representation = TreeBasedRepresentation(g, 2)

        population = [
            Individual(genotype=Leaf(1), genotype_to_phenotype=representation.genotype_to_phenotype),
            Individual(genotype=Leaf(2), genotype_to_phenotype=representation.genotype_to_phenotype),
        ]

        problem = SingleObjectiveProblem(fitness_function=lambda x: x.a, minimize=False)
        x = best_individual(population, problem)
        assert x.get_phenotype().a == 2

        problem = SingleObjectiveProblem(fitness_function=lambda x: x.a, minimize=True)
        x = best_individual(population, problem)
        assert x.get_phenotype().a == 1

        problem = MultiObjectiveProblem(minimize=[True, True], fitness_function=lambda x: [x.a, x.a])
        x = best_individual(population, problem)
        assert x.get_phenotype().a == 1

    def test_is_better(self):
        g = extract_grammar([Leaf], Root)
        representation = TreeBasedRepresentation(g, 2)
        evaluator = SequentialEvaluator()

        a = Individual(genotype=Leaf(1), genotype_to_phenotype=representation.genotype_to_phenotype)
        b = Individual(genotype=Leaf(2), genotype_to_phenotype=representation.genotype_to_phenotype)

        problem = SingleObjectiveProblem(fitness_function=lambda x: x.a, minimize=True)
        evaluator.eval(problem, [a, b])
        assert is_better(problem, a, b)

        problem = SingleObjectiveProblem(fitness_function=lambda x: x.a, minimize=False)
        evaluator.eval(problem, [a, b])
        assert not is_better(problem, a, b)

    def test_sort(self):
        g = extract_grammar([Leaf], Root)
        representation = TreeBasedRepresentation(g, 2)
        evaluator = SequentialEvaluator()

        a = Individual(genotype=Leaf(1), genotype_to_phenotype=representation.genotype_to_phenotype)
        b = Individual(genotype=Leaf(3), genotype_to_phenotype=representation.genotype_to_phenotype)
        c = Individual(genotype=Leaf(2), genotype_to_phenotype=representation.genotype_to_phenotype)
        population = [a, b, c]

        problem = SingleObjectiveProblem(fitness_function=lambda x: x.a, minimize=True)
        evaluator.eval(problem, population)
        sorted_population = sort_population(population, problem)
        assert sorted_population[0].get_phenotype().a == 1
        assert sorted_population[1].get_phenotype().a == 2
        assert sorted_population[2].get_phenotype().a == 3
