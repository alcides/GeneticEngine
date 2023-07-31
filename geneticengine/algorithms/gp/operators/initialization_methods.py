from __future__ import annotations
from typing import Any

from geneticengine.algorithms.gp.individual import Individual
from geneticengine.algorithms.gp.structure import PopulationInitializer
from geneticengine.core.problems import Problem
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import Representation
from geneticengine.core.representations.tree.initializations import full_method, grow_method
from geneticengine.core.representations.tree.initializations import pi_grow_method
from geneticengine.core.representations.tree.treebased import random_individual


class FullInitializer(PopulationInitializer):
    """All individuals are created with full trees (maximum depth in all
    branches)."""

    def __init__(self, min_init_depth=0, max_init_depth=None):
        self.min_init_depth = min_init_depth
        self.max_init_depth = max_init_depth

    def initialize(
        self,
        problem: Problem,
        representation: Representation,
        random_source: Source,
        target_size: int,
        **kwargs,
    ) -> list[Individual]:
        if not self.max_init_depth:
            self.max_init_depth = representation.max_depth
        assert self.max_init_depth <= representation.max_depth
        assert self.min_init_depth <= self.max_init_depth
        return [
            Individual(
                representation.phenotype_to_genotype(
                    random_source,
                    random_individual(
                        r=random_source,
                        g=representation.grammar,
                        max_depth=self.max_init_depth,
                        method=full_method,
                        **kwargs,
                    ),
                ),
                genotype_to_phenotype=representation.genotype_to_phenotype,
            )
            for _ in range(target_size)
        ]


class GrowInitializer(PopulationInitializer):
    """All individuals are created expanding productions until a maximum depth,
    but without the requirement of reaching that depth."""

    def __init__(self, min_init_depth=0, max_init_depth=None):
        self.min_init_depth = min_init_depth
        self.max_init_depth = max_init_depth

    def initialize(
        self,
        problem: Problem,
        representation: Representation,
        random_source: Source,
        target_size: int,
        **kwargs,
    ) -> list[Individual]:
        if not self.max_init_depth:
            self.max_init_depth == representation.max_depth
        assert self.max_init_depth <= representation.max_depth
        assert self.min_init_depth <= self.max_init_depth
        return [
            Individual(
                representation.phenotype_to_genotype(
                    random_source,
                    random_individual(
                        r=random_source,
                        g=representation.grammar,
                        max_depth=self.max_init_depth,
                        min_depth=self.min_init_depth,
                        method=grow_method,
                        **kwargs,
                    ),
                ),
                genotype_to_phenotype=representation.genotype_to_phenotype,
            )
            for _ in range(target_size)
        ]


class PositionIndependentGrowInitializer(PopulationInitializer):
    """All individuals are created expanding productions until a maximum depth,
    but without the requirement of reaching that depth."""

    def __init__(self, min_init_depth=0, max_init_depth=None):
        self.min_init_depth = min_init_depth
        self.max_init_depth = max_init_depth

    def initialize(
        self,
        problem: Problem,
        representation: Representation,
        random_source: Source,
        target_size: int,
        **kwargs,
    ) -> list[Individual]:
        if not self.max_init_depth:
            self.max_init_depth == representation.max_depth
        assert self.max_init_depth <= representation.max_depth
        assert self.min_init_depth <= self.max_init_depth
        return [
            Individual(
                representation.phenotype_to_genotype(
                    random_source,
                    random_individual(
                        r=random_source,
                        g=representation.grammar,
                        max_depth=self.max_init_depth,
                        method=pi_grow_method,
                        **kwargs,
                    ),
                ),
                genotype_to_phenotype=representation.genotype_to_phenotype,
            )
            for _ in range(target_size)
        ]


class RampedInitializer(PopulationInitializer):
    """This method uses the grow method from the minimum grammar depth to the
    maximum."""

    def __init__(self, min_init_depth=0, max_init_depth=None, method=grow_method):
        self.min_init_depth = min_init_depth
        self.max_init_depth = max_init_depth
        self.method = method

    def initialize(
        self,
        problem: Problem,
        representation: Representation,
        random_source: Source,
        target_size: int,
    ) -> list[Individual]:
        self.min_init_depth = max(representation.min_depth, self.min_init_depth)
        if not self.max_init_depth:
            self.max_init_depth = representation.max_depth
        assert self.max_init_depth <= representation.max_depth
        assert self.min_init_depth <= self.max_init_depth

        def bound(i: int):
            interval = (self.max_init_depth - self.min_init_depth) + 1
            v = self.min_init_depth + (i % interval)
            return v

        return [
            Individual(
                representation.phenotype_to_genotype(
                    random_source,
                    random_individual(
                        r=random_source,
                        g=representation.grammar,
                        max_depth=bound(i),
                        min_depth=self.min_init_depth,
                        method=self.method,
                    ),
                ),
                genotype_to_phenotype=representation.genotype_to_phenotype,
            )
            for i in range(target_size)
        ]


class RampedHalfAndHalfInitializer(PopulationInitializer):
    """Half of the individuals are created with the maximum depth, and the
    other half with different values of maximum depth between the minimum and
    the maximum.

    There's an equal chance of using full or grow method.
    """

    def __init__(self, min_init_depth=0, max_init_depth=None):
        self.min_init_depth = min_init_depth
        self.max_init_depth = max_init_depth

    def initialize(
        self,
        problem: Problem,
        representation: Representation,
        random_source: Source,
        target_size: int,
    ) -> list[Individual]:
        self.min_init_depth = max(representation.min_depth, self.min_init_depth)
        if not self.max_init_depth:
            self.max_init_depth = representation.max_depth
        assert self.max_init_depth <= representation.max_depth
        assert self.min_init_depth <= self.max_init_depth

        def bound(i: int):
            interval = (self.max_init_depth - self.min_init_depth) + 1
            v = self.min_init_depth + (i % interval)
            return v

        mid = target_size // 2
        pop = [
            Individual(
                representation.phenotype_to_genotype(
                    random_source,
                    random_individual(
                        r=random_source,
                        g=representation.grammar,
                        max_depth=bound(i),
                        min_depth=self.min_init_depth,
                        method=random_source.choice([grow_method, full_method]),
                    ),
                ),
                genotype_to_phenotype=representation.genotype_to_phenotype,
            )
            for i in range(mid)
        ] + [
            Individual(
                representation.phenotype_to_genotype(
                    random_source,
                    random_individual(
                        r=random_source,
                        g=representation.grammar,
                        max_depth=self.max_init_depth,
                        min_depth=self.min_init_depth,
                        method=random_source.choice([grow_method, full_method]),
                    ),
                ),
                genotype_to_phenotype=representation.genotype_to_phenotype,
            )
            for i in range(target_size - mid)
        ]
        return pop


class InjectInitialPopulationWrapper(PopulationInitializer):
    """Starts with an initial population, and relies on another initializer if
    it's necessary to fulfill the population size."""

    def __init__(self, programs: list[Any], backup: PopulationInitializer):
        self.programs = programs
        self.backup_initializer = backup

    def initialize(
        self,
        problem: Problem,
        representation: Representation,
        random_source: Source,
        target_size: int,
    ) -> list[Individual]:
        self.programs = [
            Individual(
                representation.phenotype_to_genotype(random_source, p1),
                genotype_to_phenotype=representation.genotype_to_phenotype,
            )
            for p1 in self.programs
        ]
        if target_size > len(self.programs):
            return self.programs[:target_size]
        else:
            return self.programs + self.backup_initializer.initialize(
                problem,
                representation,
                random_source,
                target_size - len(self.programs),
            )
