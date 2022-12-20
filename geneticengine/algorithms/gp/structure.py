from __future__ import annotations

import abc

from geneticengine.algorithms.gp.individual import Individual
from geneticengine.core.problems import Problem
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import Representation


class PopulationInitializer(abc.ABC):
    @abc.abstractmethod
    def initialize(
        self,
        p: Problem,
        representation: Representation,
        random_source: Source,
        target_size: int,
    ) -> list[Individual]:
        ...


class GeneticStep(abc.ABC):
    @abc.abstractmethod
    def iterate(
        self,
        p: Problem,
        representation: Representation,
        random_source: Source,
        population: list[Individual],
        target_size: int,
        generation: int,
    ) -> list[Individual]:
        ...

    def __str__(self):
        return f"{self.__class__.__name__}"


class StoppingCriterium(abc.ABC):
    """TerminationCondition provides information when to terminate
    evolution."""

    @abc.abstractmethod
    def is_ended(
        self,
        population: list[Individual],
        generation: int,
        elapsed_time: float,
    ) -> bool:
        ...
