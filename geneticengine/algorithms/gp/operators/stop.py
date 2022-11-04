from __future__ import annotations

from geneticengine.algorithms.gp.individual import Individual
from geneticengine.algorithms.gp.structure import StoppingCriterium


class GenerationStoppingCriterium(StoppingCriterium):
    """Runs the evolution during a number of generations."""

    def __init__(self, max_generations: int):
        """Creates a limit for the evolution, based on the number of
        generations.

        Arguments:
            max_generations (int): Number of generations to execute
        """
        self.max_generations = max_generations

    def is_ended(
        self,
        population: list[Individual],
        generation: int,
        elapsed_time: float,
    ) -> bool:
        return generation >= self.max_generations


class TimeStoppingCriterium(StoppingCriterium):
    """Runs the evolution during a given amount of time.

    Note that termination is not pre-emptive. If fitnessfunction is
    flow, this might take more than the pre-specified time.
    """

    def __init__(self, max_time: int):
        """Creates a limit for the evolution, based on the execution time.

        Arguments:
            max_time (int): Maximum time in seconds to run the evolution
        """
        self.max_time = max_time

    def is_ended(
        self,
        population: list[Individual],
        generation: int,
        elapsed_time: float,
    ) -> bool:
        return elapsed_time >= self.max_time
