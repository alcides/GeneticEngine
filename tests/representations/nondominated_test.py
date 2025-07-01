from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from geneticengine.prelude import abstract
from geneticengine.solutions.individual import PhenotypicIndividual
from geneticengine.problems import MultiObjectiveProblem, SingleObjectiveProblem
from geneticengine.grammar import extract_grammar
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.problems.helpers import non_dominated
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.random.sources import NativeRandomSource
from geneticengine.evaluation.sequential import SequentialEvaluator

@abstract
class Node:
    pass

@dataclass
class Good(Node):
    child: Node

@dataclass
class Bad(Node):
    child: Node

@dataclass
class Leaf(Node):
    pass


def multi_objective_fitness(n: Node) -> List[float]:
    good_count = 0
    bad_count = 0
    q: list[Any] = [n]
    while q:
        curr = q.pop(0)
        if isinstance(curr, Good):
            good_count += 1
        elif isinstance(curr, Bad):
            bad_count += 1

        if hasattr(curr, "__dict__"):
            for field in curr.__dict__.values():
                if isinstance(field, Node):
                    q.append(field)

    return [float(good_count), float(-bad_count)]

def single_objective_fitness(n: Node) -> float:
    """Calculates a single fitness score: good nodes minus bad nodes."""
    good_count = 0
    bad_count = 0
    q: list[Any] = [n]
    while q:
        curr = q.pop(0)
        if isinstance(curr, Good):
            good_count += 1
        elif isinstance(curr, Bad):
            bad_count += 1

        if hasattr(curr, "__dict__"):
            for field in curr.__dict__.values():
                if isinstance(field, Node):
                    q.append(field)
    return float(good_count - bad_count)

class TestNonDominated:
    def test_non_dominated_multi_objective_selection(self):
        """
        Tests if the non_dominated function correctly identifies the Pareto front
        from a hardcoded population of individuals for a multiobjective problem.
        """
        grammar = extract_grammar([Good, Bad, Leaf], Node)
        representation = TreeBasedRepresentation(grammar, decider=MaxDepthDecider(NativeRandomSource(0), grammar, max_depth=5))

        problem = MultiObjectiveProblem(
            fitness_function=multi_objective_fitness,
            minimize=[False, False],
        )

        genotypes = [
            Good(Good(Leaf())),    # Fitness: [2.0, 0.0] -> Should be in front
            Good(Bad(Leaf())),     # Fitness: [1.0, -1.0] -> Dominated by the one above
            Bad(Bad(Leaf())),      # Fitness: [0.0, -2.0] -> Dominated by all others
            Leaf(),                # Fitness: [0.0, 0.0] -> Should be in front
            Good(Leaf()),          # Fitness: [1.0, 0.0] -> Should be in front
        ]

        expected_pareto_front = [genotypes[0], genotypes[3], genotypes[4]]

        population = [PhenotypicIndividual(g, representation) for g in genotypes]

        evaluator = SequentialEvaluator()
        evaluated_population = list(evaluator.evaluate(problem, population))

        pareto_front = non_dominated(evaluated_population, problem)
        phenotypes = [ind.get_phenotype() for ind in pareto_front]

        assert all(ind in expected_pareto_front for ind in phenotypes)



    def test_non_dominated_single_objective_selection(self):
        """
        Tests if the non_dominated function correctly identifies the single best
        individual when used with a single-objective problem.
        """
        grammar = extract_grammar([Good, Bad, Leaf], Node)
        representation = TreeBasedRepresentation(grammar, decider=MaxDepthDecider(NativeRandomSource(0), grammar, max_depth=5))

        problem = SingleObjectiveProblem(
            fitness_function=single_objective_fitness,
            minimize=False,
        )

        genotypes = [
            Good(Good(Leaf())),    # Fitness: 2.0 -> The single best
            Good(Leaf()),          # Fitness: 1.0
            Leaf(),                # Fitness: 0.0
            Bad(Leaf()),           # Fitness: -1.0
            Bad(Bad(Leaf())),      # Fitness: -2.0
        ]

        population = [PhenotypicIndividual(g, representation) for g in genotypes]

        evaluator = SequentialEvaluator()
        evaluated_population = list(evaluator.evaluate(problem, population))

        pareto_front = non_dominated(evaluated_population, problem)

        phenotypes = [ind.get_phenotype() for ind in pareto_front]

        assert all(expected == actual for expected, actual in zip(genotypes, phenotypes))
