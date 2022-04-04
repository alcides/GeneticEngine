from __future__ import annotations

from typing import Callable
from typing import Type

from geneticengine.algorithms.gp.individual import Individual
from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import Representation


def create_mutation(
    r: Source,
    representation: Representation,
    g: Grammar,
    max_depth: int,
) -> Callable[[Individual], Individual]:
    def mutation(individual: Individual):
        new_individual = Individual(
            genotype=representation.mutate_individual(
                r,
                g,
                individual.genotype,
                max_depth,
                g.starting_symbol,
            ),
            fitness=None,
        )
        return new_individual

    return mutation


def create_hill_climbing_mutation(
    r: Source,
    representation: Representation,
    g: Grammar,
    max_depth: int,
    fitness_function: Callable[[Individual], float],
    n_candidates: int = 5,
) -> Callable[[Individual], Individual]:
    def hill_climbing_mutation(individual: Individual):
        def creation_new_individual():
            return Individual(
                genotype=representation.mutate_individual(
                    r,
                    g,
                    individual.genotype,
                    max_depth,
                    g.starting_symbol,
                ),
                fitness=None,
            )

        new_individuals = [creation_new_individual() for _ in range(n_candidates)]
        best_individual = min(
            (new_individuals + [individual]),
            key=fitness_function,
        )
        return best_individual

    return hill_climbing_mutation
