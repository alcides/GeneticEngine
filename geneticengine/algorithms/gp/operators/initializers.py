from __future__ import annotations
from typing import Iterator

from geneticengine.solutions.individual import PhenotypicIndividual
from geneticengine.algorithms.gp.structure import PopulationInitializer
from geneticengine.problems import Problem
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import Representation


class HalfAndHalfInitializer(PopulationInitializer):
    """Combines two initializers, one for each half of the population."""

    def __init__(self, initializer1, initializer2):
        self.initializer1 = initializer1
        self.initializer2 = initializer2

    def initialize(
        self,
        problem: Problem,
        representation: Representation,
        random: RandomSource,
        target_size: int,
        **kwargs,
    ) -> Iterator[PhenotypicIndividual]:
        mid = target_size // 2
        yield from self.initializer1(problem, representation, random, mid)
        yield from self.initializer2(
            problem,
            representation,
            random,
            target_size - mid,
        )


class StandardInitializer(PopulationInitializer):
    """All individuals are created with full trees (maximum depth in all
    branches)."""

    def initialize(
        self,
        problem: Problem,
        representation: Representation,
        random: RandomSource,
        target_size: int,
        **kwargs,
    ) -> Iterator[PhenotypicIndividual]:
        for i in range(target_size):
            yield PhenotypicIndividual(
                representation.create_genotype(
                    random,
                    **kwargs,
                ),
                representation=representation,
            )
