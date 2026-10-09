from dataclasses import dataclass

from geneticengine.algorithms.gp.operators.selection import BloatControlledTournamentSelection
from geneticengine.grammar.decorators import abstract
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.solutions.individual import PhenotypicIndividual


@abstract
class Node:
    pass


@dataclass
class Leaf(Node):
    pass


@dataclass
class Branch(Node):
    child: Node


def test_dynamic_bloat_limit_is_optional_and_applied():
    random = NativeRandomSource(2)
    grammar = extract_grammar([Leaf, Branch], Node)
    representation = TreeBasedRepresentation(grammar, MaxDepthDecider(random, grammar, 4))
    small = representation.create_genotype(random, decider=MaxDepthDecider(random, grammar, 1))
    large = representation.create_genotype(random, decider=MaxDepthDecider(random, grammar, 4))
    problem = SingleObjectiveProblem(lambda node: 1.0, minimize=True)
    individuals = [PhenotypicIndividual(small, representation), PhenotypicIndividual(large, representation)]
    for individual in individuals:
        individual.set_fitness(problem, problem.evaluate(individual.get_phenotype()))
    selected = list(
        BloatControlledTournamentSelection(1, max_nodes=lambda generation: 1).iterate(
            problem, _Evaluator(individuals), representation, random, individuals, 2, 0,
        ),
    )
    assert all(ind.get_phenotype().gengy_nodes <= 1 for ind in selected)


class _Evaluator:
    def __init__(self, individuals):
        self.individuals = individuals

    def evaluate(self, problem, population):
        return self.individuals
