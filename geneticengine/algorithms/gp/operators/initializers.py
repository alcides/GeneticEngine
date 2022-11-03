from __future__ import annotations

from typing import Any

from geneticengine.algorithms.gp.individual import Individual
from geneticengine.algorithms.gp.structure import PopulationInitializer
from geneticengine.core.problems import Problem
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import Representation


class FullInitializer(PopulationInitializer):
    """All individuals are created with full trees."""

    def initialize(
        self,
        p: Problem,
        representation: Representation,
        random_source: Source,
        target_size: int,
    ) -> list[Individual]:
        # TODO Mode: full, grow
        return [
            Individual(
                representation.create_individual(random_source),
                genotype_to_phenotype=representation.genotype_to_phenotype,
            )
            for _ in range(target_size)
        ]


class GrowInitializer(PopulationInitializer):
    """All individuals are created with full trees."""

    def initialize(
        self,
        p: Problem,
        representation: Representation,
        random_source: Source,
        target_size: int,
    ) -> list[Individual]:
        # TODO Mode: full, grow
        return [
            Individual(
                representation.create_individual(
                    random_source,
                ),
                genotype_to_phenotype=representation.genotype_to_phenotype,
            )
            for _ in range(target_size)
        ]


class RampedHalfAndHalfInitializer(PopulationInitializer):
    """All individuals are created with full trees."""

    def initialize(
        self,
        p: Problem,
        representation: Representation,
        random_source: Source,
        target_size: int,
    ) -> list[Individual]:
        # TODO Mode: full, grow
        return [
            Individual(
                representation.create_individual(
                    random_source,
                    random_source.randint(
                        representation.min_depth,
                        representation.max_depth,
                    ),
                ),
                genotype_to_phenotype=representation.genotype_to_phenotype,
            )
            for _ in range(target_size)
        ]


class InjectInitialPopulationWrapper(PopulationInitializer):
    """Starts with an initial population, and relies on another initializer is
    necessary to fulfill the population size."""

    def __init__(self, programs: list[Any], backup: PopulationInitializer):
        self.programs = programs
        self.backup_initializer = backup

    def initialize(
        self,
        p: Problem,
        representation: Representation,
        random_source: Source,
        target_size: int,
    ) -> list[Individual]:
        self.programs = [
            Individual(p, genotype_to_phenotype=representation.genotype_to_phenotype)
            for p in self.programs
        ]
        if target_size > len(self.programs):
            return self.programs[:target_size]
        else:
            return self.programs + self.backup_initializer.initialize(
                p,
                representation,
                random_source,
                target_size - len(self.programs),
            )
