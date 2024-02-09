from __future__ import annotations

import abc
from geneticengine.representations.api import SolutionRepresentation

from geneticengine.solutions.individual import Individual
from geneticengine.problems import Problem
from geneticengine.random.sources import RandomSource
from geneticengine.evaluation import Evaluator


class PopulationInitializer(abc.ABC):
    @abc.abstractmethod
    def initialize(
        self,
        problem: Problem,
        representation: SolutionRepresentation,
        random_source: RandomSource,
        target_size: int,
    ) -> list[Individual]:
        ...


class GeneticStep(abc.ABC):
    @abc.abstractmethod
    def iterate(
        self,
        problem: Problem,
        evaluator: Evaluator,
        representation: SolutionRepresentation,
        random_source: RandomSource,
        population: list[Individual],
        target_size: int,
        generation: int,
    ) -> list[Individual]:
        ...

    def __str__(self):
        return f"{self.__class__.__name__}"
