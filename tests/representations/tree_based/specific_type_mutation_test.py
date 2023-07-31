from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated
from geneticengine.algorithms.gp.gp import GP
from geneticengine.algorithms.gp.operators.combinators import ParallelStep, SequenceStep
from geneticengine.algorithms.gp.operators.crossover import GenericCrossoverStep
from geneticengine.algorithms.gp.operators.elitism import ElitismStep
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.algorithms.gp.operators.selection import TournamentSelection
from geneticengine.algorithms.gp.operators.stop import GenerationStoppingCriterium

from geneticengine.core.decorators import abstract
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.representations.tree.treebased import TreeBasedRepresentation, TypeSpecificTBMutation
from geneticengine.metahandlers.lists import ListSizeBetween


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
    "probability_crossover" : .9,
    "probability_mutation" : 1,
    "number_of_generations" : 5,
    "max_depth" : 10,
    "population_size" : 2,
    "tournament_size" : 2,
    "n_elites" : 1,
}
def algorithm_steps():
    """The default step in Genetic Programming."""
    return ParallelStep(
        [
            ElitismStep(),
            SequenceStep(
                TournamentSelection(gp_parameters["tournament_size"]),
                GenericCrossoverStep(gp_parameters["probability_crossover"]),
                GenericMutationStep(gp_parameters["probability_mutation"], operator=TypeSpecificTBMutation(Concrete)),
            ),
        ],
        weights=[gp_parameters["n_elites"], 100 - gp_parameters["n_elites"]],
    )

def fitness_function(x : Root):
    return 1
problem = SingleObjectiveProblem(minimize=False, fitness_function=fitness_function)

class TestNodesDepthSpecific:
    def test_nodes_depth_specific_simple(self):
        g = extract_grammar([Concrete, Middle, MiddleList, ConcreteTerm, RootToConcrete], Root)
        alg = GP(
            representation=TreeBasedRepresentation(g, gp_parameters["max_depth"]),
            problem=problem,
            step=algorithm_steps(),
            stopping_criterium=GenerationStoppingCriterium(gp_parameters["number_of_generations"]),
            population_size=gp_parameters["population_size"],
        )
        ind = alg.evolve()
        assert ind
