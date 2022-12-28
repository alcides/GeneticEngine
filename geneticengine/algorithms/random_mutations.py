from __future__ import annotations

from typing import Any

from geneticengine.algorithms.callbacks.callback import Callback
from geneticengine.algorithms.gp.gp import GP
from geneticengine.algorithms.gp.operators.combinators import ParallelStep
from geneticengine.algorithms.gp.operators.combinators import SequenceStep
from geneticengine.algorithms.gp.operators.elitism import ElitismStep
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.algorithms.gp.structure import PopulationInitializer
from geneticengine.algorithms.gp.structure import StoppingCriterium
from geneticengine.core.problems import Problem
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import Representation


class RandomMutations(GP):
    """Random Mutations is a GP instance whose main step is to mutate the
    current individual, and keep only the best of the two.

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
        random_source: Source,
        initializer: PopulationInitializer,
        stopping_criterium: StoppingCriterium,
        callbacks: list[Callback] | None = None,
    ):
        callbacks = callbacks or []
        step = SequenceStep(ParallelStep([GenericMutationStep(1), ElitismStep()]))
        super().__init__(
            representation,
            problem,
            random_source,
            2,
            initializer,
            step,
            stopping_criterium,
            callbacks,
        )
