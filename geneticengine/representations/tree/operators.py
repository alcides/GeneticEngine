from __future__ import annotations
from typing import Any

from geneticengine.solutions.individual import Individual
from geneticengine.algorithms.gp.structure import PopulationInitializer
from geneticengine.problems import Problem
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import SolutionRepresentation
from geneticengine.representations.tree.treebased import TreeBasedRepresentation

# TODO: Redo initialization to be parameterless


class FullInitializer(PopulationInitializer):
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
        assert isinstance(representation, TreeBasedRepresentation)
        return [
            Individual(
                representation.instantiate(
                    random_source,
                ),
                genotype_to_phenotype=representation.map,
            )
            for _ in range(target_size)
        ]


class GrowInitializer(PopulationInitializer):
    """All individuals are created expanding productions until a maximum depth,
    but without the requirement of reaching that depth."""

    def initialize(
        self,
        problem: Problem,
        representation: SolutionRepresentation,
        random_source: RandomSource,
        target_size: int,
        **kwargs,
    ) -> list[Individual]:
        assert isinstance(representation, TreeBasedRepresentation)
        return [
            Individual(
                representation.instantiate(
                    random_source,
                ),
                genotype_to_phenotype=representation.map,
            )
            for _ in range(target_size)
        ]


class PositionIndependentGrowInitializer(PopulationInitializer):
    """All individuals are created expanding productions until a maximum depth,
    but without the requirement of reaching that depth."""

    def initialize(
        self,
        problem: Problem,
        representation: SolutionRepresentation,
        random_source: RandomSource,
        target_size: int,
        **kwargs,
    ) -> list[Individual]:
        assert isinstance(representation, TreeBasedRepresentation)
        return [
            Individual(
                representation.instantiate(
                    random_source,
                ),
                genotype_to_phenotype=representation.map,
            )
            for _ in range(target_size)
        ]


class RampedInitializer(PopulationInitializer):
    """This method uses the grow method from the minimum grammar depth to the
    maximum."""

    def initialize(
        self,
        problem: Problem,
        representation: SolutionRepresentation,
        random_source: RandomSource,
        target_size: int,
    ) -> list[Individual]:
        assert isinstance(representation, TreeBasedRepresentation)
        return [
            Individual(
                representation.instantiate(
                    random_source,
                ),
                genotype_to_phenotype=representation.map,
            )
            for _ in range(target_size)
        ]


class RampedHalfAndHalfInitializer(PopulationInitializer):
    """Half of the individuals are created with the maximum depth, and the
    other half with different values of maximum depth between the minimum and
    the maximum.

    There's an equal chance of using full or grow method.
    """

    def initialize(
        self,
        problem: Problem,
        representation: SolutionRepresentation,
        random_source: RandomSource,
        target_size: int,
    ) -> list[Individual]:
        assert isinstance(representation, TreeBasedRepresentation)
        return [
            Individual(
                representation.instantiate(
                    random_source,
                ),
                genotype_to_phenotype=representation.map,
            )
            for _ in range(target_size)
        ]


class InjectInitialPopulationWrapper(PopulationInitializer):
    """Starts with an initial population, and relies on another initializer if
    it's necessary to fulfill the population size."""

    def __init__(self, programs: list[Any], backup: PopulationInitializer):
        self.programs = programs
        self.backup_initializer = backup

    def initialize(
        self,
        problem: Problem,
        representation: SolutionRepresentation,
        random_source: RandomSource,
        target_size: int,
    ) -> list[Individual]:
        assert isinstance(representation, TreeBasedRepresentation)
        self.programs = [Individual(p1, genotype_to_phenotype=representation.map) for p1 in self.programs]
        if target_size > len(self.programs):
            return self.programs[:target_size]
        else:
            return self.programs + self.backup_initializer.initialize(
                problem,
                representation,
                random_source,
                target_size - len(self.programs),
            )
