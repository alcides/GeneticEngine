from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.algorithms.gp.operators.combinators import ParallelStep, SequenceStep
from geneticengine.algorithms.gp.operators.crossover import GenericCrossoverStep
from geneticengine.algorithms.gp.operators.elitism import ElitismStep
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.algorithms.gp.operators.selection import TournamentSelection
from geneticengine.evaluation.budget import EvaluationBudget

from geneticengine.grammar.decorators import abstract
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.grammar.metahandlers.lists import ListSizeBetween


@abstract
class Root:
    pass


@dataclass
class Middle(Root):
    child: Root


@abstract
class Concrete:
    pass


@dataclass
class RootToConcrete(Root):
    a: Concrete
    b: Concrete


class ConcreteTerm(Concrete):
    pass


@dataclass
class MiddleList(Root):
    z: Annotated[list[Root], ListSizeBetween(2, 3)]


gp_parameters = {
    "crossover_probability": 0.9,
    "mutation_probability": 1,
    "number_of_generations": 5,
    "max_depth": 10,
    "population_size": 2,
    "tournament_size": 2,
    "elitism": 1,
}


def algorithm_steps():
    """The default step in Genetic Programming."""
    return ParallelStep(
        [
            ElitismStep(),
            SequenceStep(
                TournamentSelection(gp_parameters["tournament_size"]),
                GenericCrossoverStep(gp_parameters["crossover_probability"]),
                GenericMutationStep(gp_parameters["mutation_probability"]),
            ),
        ],
        weights=[gp_parameters["elitism"], 100 - gp_parameters["elitism"]],
    )


def fitness_function(x: Root):
    return 1


problem = SingleObjectiveProblem(minimize=False, fitness_function=fitness_function)


class TestNodesDepthSpecific:
    def test_nodes_depth_specific_simple(self):
        g = extract_grammar([Concrete, Middle, MiddleList, ConcreteTerm, RootToConcrete], Root)
        r = NativeRandomSource(seed=3)
        alg = GeneticProgramming(
            representation=TreeBasedRepresentation(g, MaxDepthDecider(r, g, gp_parameters["max_depth"])),
            budget=EvaluationBudget(100),
            problem=problem,
            random=r,
            step=algorithm_steps(),
            population_size=gp_parameters["population_size"],
        )
        ind = alg.search()[0]
        assert ind
