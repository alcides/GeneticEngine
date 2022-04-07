from __future__ import annotations

from typing import Callable
from typing import Tuple

from geneticengine.algorithms.gp.individual import Individual
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import Representation
from geneticengine.core.representations.tree.treebased import Grammar


def create_cross_over(
    r: Source,
    representation: Representation,
    g: Grammar,
    max_depth: int,
) -> Callable[[Individual, Individual], tuple[Individual, Individual]]:
    def cross_over_double(
        individual1: Individual,
        individual2: Individual,
    ) -> tuple[Individual, Individual]:
        (g1, g2) = representation.crossover_individuals(
            r,
            g,
            individual1.genotype,
            individual2.genotype,
            max_depth,
        )
        individual1 = Individual(g1)
        individual2 = Individual(g2)
        return (individual1, individual2)

    return cross_over_double
