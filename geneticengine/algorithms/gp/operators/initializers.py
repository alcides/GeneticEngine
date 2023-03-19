from __future__ import annotations

from geneticengine.algorithms.gp.individual import Individual
from geneticengine.algorithms.gp.structure import PopulationInitializer
from geneticengine.core.problems import Problem
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import Representation


class HalfAndHalfInitializer(PopulationInitializer):
    """Combines two initializers, one for each half of the population."""

    def __init__(self, initializer1, initializer2):
        self.initializer1 = initializer1
        self.initializer2 = initializer2

    def initialize(
        self,
        problem: Problem,
        representation: Representation,
        random_source: Source,
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
        representation: Representation,
        random_source: Source,
        target_size: int,
        **kwargs,
    ) -> list[Individual]:
        return [
            Individual(
                representation.create_individual(
                    random_source,
                    representation.max_depth,
                    **kwargs,
                ),
                genotype_to_phenotype=representation.genotype_to_phenotype,
            )
            for _ in range(target_size)
        ]
