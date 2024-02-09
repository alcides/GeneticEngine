from __future__ import annotations

from geneticengine.solutions.individual import Individual
from geneticengine.algorithms.gp.structure import PopulationInitializer
from geneticengine.problems import Problem
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import SolutionRepresentation


class HalfAndHalfInitializer(PopulationInitializer):
    """Combines two initializers, one for each half of the population."""

    def __init__(self, initializer1, initializer2):
        self.initializer1 = initializer1
        self.initializer2 = initializer2

    def initialize(
        self,
        problem: Problem,
        representation: SolutionRepresentation,
        random_source: RandomSource,
        target_size: int,
        **kwargs,
    ) -> list[Individual]:
        mid = target_size // 2
        return self.initializer1(problem, representation, random_source, mid) + self.initializer2(
            problem,
            representation,
            random_source,
            target_size - mid,
        )


class StandardInitializer(PopulationInitializer):
    """All individuals are created with full trees (maximum depth in all
    branches)."""

    def initialize(
        self,
        problem: Problem,
        representation: SolutionRepresentation,
        random_source: RandomSource,
        target_size: int,
        **kwargs,
    ) -> list[Individual]:
        return [
            Individual(
                representation.instantiate(
                    random_source,
                    **kwargs,
                ),
                genotype_to_phenotype=representation.map,
            )
            for _ in range(target_size)
        ]
