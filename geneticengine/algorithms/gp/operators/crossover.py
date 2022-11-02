from __future__ import annotations

from typing import Callable
from typing import Optional
from typing import Tuple

from geneticengine.algorithms.gp.individual import Individual
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.core.problems import Problem
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import Representation
from geneticengine.core.representations.tree.treebased import Grammar


class GenericCrossoverStep(GeneticStep):
    def __init__(
        self,
        probability: float,
        specific_type: type | None = None,
        depth_aware_co: bool = False,
    ):
        self.probability = probability
        self.specific_type = specific_type
        self.depth_aware_co = depth_aware_co

    def iterate(
        self,
        p: Problem,
        representation: Representation,
        random_source: Source,
        population: list[Individual],
        target_size: int,
    ) -> list[Individual]:
        assert len(population) == target_size
        mid = len(population) // 2
        retlist = []
        for (ind1, ind2) in zip(population[:mid], population[mid:]):
            (n1, n2) = self.crossover(ind1, ind2, representation, random_source)
            retlist.append(n1)
            retlist.append(n2)
        return retlist

    def crossover(
        self,
        individual1: Individual,
        individual2: Individual,
        representation: Representation,
        random_source: Source,
    ):
        (g1, g2) = representation.crossover_individuals(
            random_source,
            individual1.genotype,
            individual2.genotype,
            representation.max_depth,
            specific_type=self.specific_type,
            depth_aware_co=self.depth_aware_co,
        )
        return (Individual(g1), Individual(g2))
