from __future__ import annotations

import abc
from typing import Generic
from typing import TypeVar

from geneticengine.random.sources import RandomSource

g = TypeVar("g")
p = TypeVar("p")


class Representation(Generic[g, p]):
    @abc.abstractmethod
    def instantiate(self, random: RandomSource, **kwargs) -> g:
        ...

    @abc.abstractmethod
    def map(self, internal: g) -> p:
        ...


class RepresentationWithMutation(Generic[g]):
    @abc.abstractmethod
    def mutate(self, random: RandomSource, internal: g) -> g:
        ...


class RepresentationWithCrossover(Generic[g]):
    @abc.abstractmethod
    def crossover(self, random: RandomSource, parent1: g, parent2: g) -> tuple[g, g]:
        ...
