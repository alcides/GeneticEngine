from typing import Callable
from geneticengine.algorithms.gp.Individual import Individual
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.api import Representation
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


def create_hill_climbing_mutation(
    r: RandomSource,
    representation: Representation,
    g: Grammar,
    max_depth: int,
    fitness_function: Callable[[Individual], float],
    n_candidates: int = 5,
) -> Callable[[Individual], Individual]:
    def hill_climbing_mutation(individual: Individual):
        creation_new_individual = lambda : Individual(
                genotype=representation.mutate_individual(
                    r, g, individual.genotype, max_depth
                ),
                fitness=None,
            )
        new_individuals = [creation_new_individual() for _ in range(n_candidates)]
        best_individual = min((new_individuals + [individual]), key=fitness_function)
        return best_individual

    return hill_climbing_mutation