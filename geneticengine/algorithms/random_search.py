from __future__ import annotations

from typing import Any

from geneticengine.algorithms.callbacks.callback import Callback
from geneticengine.algorithms.gp.gp import GP
from geneticengine.algorithms.gp.operators.combinators import ParallelStep
from geneticengine.algorithms.gp.operators.elitism import ElitismStep
from geneticengine.algorithms.gp.operators.novelty import NoveltyStep
from geneticengine.algorithms.gp.operators.stop import GenerationStoppingCriterium
from geneticengine.algorithms.gp.structure import PopulationInitializer
from geneticengine.algorithms.gp.structure import StoppingCriterium
from geneticengine.core.problems import Problem
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import Representation
from geneticengine.core.representations.tree.operators import GrowInitializer


class RandomSearch(GP):
    """Random Search is a GP instance whose main step is novelty with an
    elitism of 1.

    Args:
        representation (Representation): The individual representation used by the GP program.
        problem (Problem): A SingleObjectiveProblem or a MultiObjectiveProblem problem.
        random_source (RandomSource]): A RNG instance
        initializer (PopulationInitializer): The method to generate new individuals.
        stopping_criterium (StoppingCriterium): The class that defines how the evolution stops.
        callbacks (List[Callback]): The callbacks to define what is done with the returned prints from the algorithm
            (default = None).
    """

    def __init__(
        self,
        representation: Representation[Any, Any],
        problem: Problem,
        random_source: Source = RandomSource(0),
        initializer: PopulationInitializer = GrowInitializer(),
        stopping_criterium: StoppingCriterium = GenerationStoppingCriterium(100),
        callbacks: list[Callback] | None = None,
    ):
        concrente_callbacks: list[Callback] = callbacks if callbacks else []
        step = ParallelStep([NoveltyStep(), ElitismStep()])
        super().__init__(
            representation,
            problem,
            random_source,
            2,
            initializer,
            step,
            stopping_criterium,
            concrente_callbacks,
        )
