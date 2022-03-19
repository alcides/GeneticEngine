from __future__ import annotations

from copy import deepcopy
from typing import Callable
from typing import List

from geneticengine.algorithms.gp.individual import Individual
from geneticengine.core.random.sources import RandomSource


def create_tournament(
    tournament_size: int,
    minimize=False,
) -> Callable[[RandomSource, list[Individual], int], list[Individual]]:
    def tournament(
        r: RandomSource,
        population: list[Individual],
        n_winners: int,
    ) -> list[Individual]:
        winners = []
        for _ in range(n_winners):
            candidates = [r.choice(population) for _ in range(tournament_size)]
            winner = candidates[0]
            assert winner.fitness is not None
            for o in candidates[1:]:
                assert o.fitness is not None
                if (o.fitness > winner.fitness and not minimize) or (
                    o.fitness < winner.fitness and minimize
                ):
                    winner = o
            winners.append(winner)
            candidates.remove(winner)
        return winners

    return tournament


def create_elitism(
    n_elites: int,
) -> Callable[[list[Individual], Callable[[Individual], float]], list[Individual]]:
    def elitism(
        population: list[Individual],
        fitness: Callable[[Individual], float],
    ) -> list[Individual]:
        population = sorted(population, key=fitness)
        return [x for x in population[:n_elites]]

    return elitism


def create_novelties(
    create_individual: Callable[[int], Individual],
    max_depth: int,
) -> Callable[[int], list[Individual]]:
    def novelties(n_novelties: int) -> list[Individual]:
        return [create_individual(max_depth) for _ in range(n_novelties)]

    return novelties
