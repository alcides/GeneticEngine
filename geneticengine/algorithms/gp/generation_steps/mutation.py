from typing import Callable
from geneticengine.algorithms.gp.Individual import Individual
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.base import Representation
from geneticengine.core.grammar import Grammar


def create_mutation(
    r: RandomSource,
    representation: Representation,
    g: Grammar,
    max_depth: int,
) -> Callable[[Individual], Individual]:
    def mutation(individual: Individual):
        new_individual = Individual(
            genotype=representation.mutate_individual(
                r, g, individual.genotype, max_depth
            ),
            fitness=None,
        )
        return new_individual

    return mutation