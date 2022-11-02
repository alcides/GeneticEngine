from __future__ import annotations

from abc import ABC

from geneticengine.algorithms.gp.individual import Individual
from geneticengine.core.problems import Problem
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import Representation


class PopulationInitializer(ABC):
    def initialize(
        self,
        p: Problem,
        representation: Representation,
        random_source: Source,
        target_size: int,
    ) -> list[Individual]:
        ...


class GeneticStep(ABC):
    def iterate(
        self,
        p: Problem,
        representation: Representation,
        random_source: Source,
        population: list[Individual],
        target_size: int,
    ) -> list[Individual]:
        ...


class StoppingCriterium(ABC):
    """TerminationCondition provides information when to terminate
    evolution."""

    def is_ended(
        self,
        population: list[Individual],
        generation: int,
        elapsed_time: float,
    ) -> bool:
        ...
