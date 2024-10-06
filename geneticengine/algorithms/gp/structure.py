from __future__ import annotations

import abc
from typing import Iterator
from geneticengine.representations.api import Representation

from geneticengine.solutions.individual import Individual
from geneticengine.problems import Problem
from geneticengine.random.sources import RandomSource
from geneticengine.evaluation import Evaluator


class PopulationInitializer(abc.ABC):
    @abc.abstractmethod
    def initialize(
        self,
        problem: Problem,
        representation: Representation,
        random: RandomSource,
        target_size: int,
    ) -> Iterator[Individual]: ...


class GeneticStep(abc.ABC):
    @abc.abstractmethod
    def iterate(
        self,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random: RandomSource,
        population: Iterator[Individual],
        target_size: int,
        generation: int,
    ) -> Iterator[Individual]: ...

    def apply(
        self,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random: RandomSource,
        population: Iterator[Individual],
        target_size: int,
        generation: int,
    ) -> Iterator[Individual]:
        self.pre_iterate(problem, evaluator, representation, random, population, target_size, generation)
        current = []
        for ind in self.iterate(problem, evaluator, representation, random, population, target_size, generation):
            current.append(ind)
            yield ind
        self.post_iterate(problem, evaluator, representation, random, (i for i in current), target_size, generation)

    def pre_iterate(
        self,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random: RandomSource,
        population: Iterator[Individual],
        target_size: int,
        generation: int,
    ) -> None:
        pass

    def post_iterate(
        self,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random: RandomSource,
        population: Iterator[Individual],
        target_size: int,
        generation: int,
    ) -> None:
        pass

    def __str__(self):
        return f"{self.__class__.__name__}"
