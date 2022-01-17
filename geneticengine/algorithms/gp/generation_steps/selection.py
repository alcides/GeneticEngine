from typing import List, Callable
from copy import deepcopy
from geneticengine.core.random.sources import RandomSource
from geneticengine.algorithms.gp.individual import Individual


def create_tournament(
    tournament_size: int, minimize=False
) -> Callable[[RandomSource, List[Individual], int], List[Individual]]:
    def tournament(
        r: RandomSource, population: List[Individual], n_winners: int
    ) -> List[Individual]:
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
) -> Callable[[List[Individual], Callable[[Individual], float]], List[Individual]]:
    def elitism(
        population: List[Individual], fitness: Callable[[Individual], float]
    ) -> List[Individual]:
        population = sorted(population, key=fitness)
        return [deepcopy(x) for x in population[:n_elites]]

    return elitism


def create_novelties(
    create_individual: Callable[[int], Individual], max_depth: int
) -> Callable[[int], List[Individual]]:
    def novelties(n_novelties: int) -> List[Individual]:
        return [create_individual(max_depth) for _ in range(n_novelties)]

    return novelties
