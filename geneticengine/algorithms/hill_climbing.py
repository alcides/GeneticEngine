from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Type

from geneticengine.algorithms.callbacks.callback import Callback
from geneticengine.algorithms.gp.gp import GP
from geneticengine.algorithms.gp.individual import Individual
from geneticengine.algorithms.gp.operators.combinators import ParallelStep
from geneticengine.algorithms.gp.operators.combinators import SequenceStep
from geneticengine.algorithms.gp.operators.elitism import ElitismStep
from geneticengine.algorithms.gp.operators.initializers import (
    GrowInitializer,
)
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.algorithms.gp.operators.mutation import HillClimbingMutationIteration
from geneticengine.algorithms.gp.operators.selection import TournamentSelection
from geneticengine.algorithms.gp.operators.stop import GenerationStoppingCriterium
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.algorithms.gp.structure import PopulationInitializer
from geneticengine.algorithms.gp.structure import StoppingCriterium
from geneticengine.algorithms.heuristics import Heuristics
from geneticengine.core.grammar import Grammar
from geneticengine.core.problems import FitnessType
from geneticengine.core.problems import Problem
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import Representation


class HC(GP):
    """Hill Climbing is a GP instance whose main step is to mutate the current
    individual, and keep only the best of the two.

    Args:
        representation (Representation): The individual representation used by the GP program.
        problem (Problem): A SingleObjectiveProblem or a MultiObjectiveProblem problem.
        random_source (RandomSource]): A RNG instance
        initializer (PopulationInitializer): The method to generate new individuals.
        stopping_criterium (StoppingCriterium): The class that defines how the evolution stops.
        callbacks (List[Callback]): The callbacks to define what is done with the returned prints from the algorithm (default = []).
    """

    def __init__(
        self,
        representation: Representation[Any, Any],
        problem: Problem,
        random_source: Source = RandomSource(0),
        initializer: PopulationInitializer = GrowInitializer(),
        stopping_criterium: StoppingCriterium = GenerationStoppingCriterium(100),
        callbacks: list[Callback] = None,
    ):
        step = HillClimbingMutationIteration(1)
        GP.__init__(
            self,
            representation,
            problem,
            random_source,
            1,
            initializer,
            step,
            stopping_criterium,
            callbacks,
        )
