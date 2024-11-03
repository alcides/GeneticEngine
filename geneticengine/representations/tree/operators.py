from __future__ import annotations
from typing import Iterator

from geneticengine.exceptions import GeneticEngineError
from geneticengine.representations.tree.initializations import (
    FullDecider,
    MaxDepthDecider,
)
from geneticengine.solutions.individual import Individual, PhenotypicIndividual
from geneticengine.algorithms.gp.structure import PopulationInitializer
from geneticengine.problems import Problem
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import Representation
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.solutions.tree import TreeNode


class FullInitializer(PopulationInitializer):
    """All individuals are created with full trees (maximum depth in all
    branches)."""

    def __init__(self, max_depth: int):
        self.max_depth = max_depth

    def initialize(
        self,
        problem: Problem,
        representation: Representation,
        random: RandomSource,
        target_size: int,
        **kwargs,
    ) -> Iterator[PhenotypicIndividual]:
        assert isinstance(representation, TreeBasedRepresentation)
        for _ in range(target_size):
            yield PhenotypicIndividual(
                representation.create_genotype(
                    random,
                    decider=FullDecider(random, representation.grammar, max_depth=self.max_depth + 1),
                ),
                representation=representation,
            )


class GrowInitializer(PopulationInitializer):
    """All individuals are created expanding productions until a maximum depth,
    but without the requirement of reaching that depth."""

    def __init__(self):
        pass

    def initialize(
        self,
        problem: Problem,
        representation: Representation,
        random: RandomSource,
        target_size: int,
        max_tries: int = 1000,
        **kwargs,
    ) -> Iterator[PhenotypicIndividual]:
        assert isinstance(representation, TreeBasedRepresentation)
        generated = 0
        current_failures = 0
        depth = 1
        while generated < target_size:
            try:
                yield PhenotypicIndividual(
                    representation.create_genotype(
                        random,
                        decider=MaxDepthDecider(random, representation.grammar, max_depth=depth),
                    ),
                    representation=representation,
                )
                generated += 1
                current_failures = 0
            except GeneticEngineError:
                current_failures += 1
                if current_failures >= max_tries:
                    depth += 1


class PositionIndependentGrowInitializer(PopulationInitializer):
    """All individuals are created expanding productions until a maximum depth,
    but without the requirement of reaching that depth."""

    def __init__(self, max_depth: int):
        self.max_depth = max_depth
        self.grow = GrowInitializer()
        self.full = FullInitializer(max_depth)

    def initialize(
        self,
        problem: Problem,
        representation: Representation,
        random: RandomSource,
        target_size: int,
        **kwargs,
    ) -> Iterator[PhenotypicIndividual]:
        assert isinstance(representation, TreeBasedRepresentation)
        half = target_size // 2
        yield from self.grow.initialize(problem, representation, random, half)
        yield from self.full.initialize(problem, representation, random, target_size - half)


class RampedHalfAndHalfInitializer(PopulationInitializer):
    """Half of the individuals are created with the maximum depth, and the
    other half with different values of maximum depth between the minimum and
    the maximum.

    There's an equal chance of using full or grow method.
    """

    def __init__(self, max_depth: int):
        self.max_depth = max_depth

    def initialize(
        self,
        problem: Problem,
        representation: Representation,
        random: RandomSource,
        target_size: int,
    ) -> Iterator[PhenotypicIndividual]:
        assert isinstance(representation, TreeBasedRepresentation)
        for _ in range(target_size):

            yield PhenotypicIndividual(
                representation.create_genotype(
                    random,
                ),
                representation=representation,
            )


class InjectInitialPopulationWrapper(PopulationInitializer):
    """Starts with an initial population, and relies on another initializer if
    it's necessary to fulfill the population size."""

    def __init__(self, programs: list[TreeNode | Individual], backup: PopulationInitializer):
        self.programs = programs
        self.backup_initializer = backup

    def initialize(
        self,
        problem: Problem,
        representation: Representation,
        random: RandomSource,
        target_size: int,
    ) -> Iterator[PhenotypicIndividual]:
        assert isinstance(representation, TreeBasedRepresentation)

        def ensure_ind(x):
            if isinstance(x, Individual):
                return x
            else:
                return PhenotypicIndividual(x, representation=representation)

        for i, p in enumerate(self.programs[:target_size]):
            yield ensure_ind(p)

        if i < target_size - 1:
            yield from self.backup_initializer.initialize(problem, representation, random, target_size - i)
