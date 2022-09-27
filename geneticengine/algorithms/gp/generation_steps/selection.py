from __future__ import annotations

from copy import deepcopy
from typing import Callable
from typing import List

from geneticengine.algorithms.gp.individual import Individual
from geneticengine.core.problems import MultiObjectiveProblem
from geneticengine.core.problems import Problem
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.random.sources import RandomSource


def create_tournament(
    tournament_size: int,
    problem: Problem,
) -> Callable[[RandomSource, list[Individual], int], list[Individual]]:
    assert isinstance(problem, SingleObjectiveProblem)

    def tournament(
        r: RandomSource,
        population: list[Individual],
        n_winners: int,
    ) -> list[Individual]:
        assert isinstance(problem, SingleObjectiveProblem)
        winners = []
        for _ in range(n_winners):
            candidates = [r.choice(population) for _ in range(tournament_size)]
            winner = candidates[0]
            assert winner.fitness is not None
            assert isinstance(winner.fitness, float)
            for o in candidates[1:]:
                assert o.fitness is not None
                assert isinstance(o.fitness, float)
                if (o.fitness > winner.fitness and not problem.minimize) or (
                    o.fitness < winner.fitness and problem.minimize
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
        elites: list[Individual] = list()
        i = 0
        while len(elites) < n_elites and i < len(population):
            if population[i] not in elites:
                elites.append(population[i])
            i += 1
        return elites

    return elitism


def create_novelties(
    create_individual: Callable[[int], Individual],
    max_depth: int,
) -> Callable[[int], list[Individual]]:
    def novelties(n_novelties: int) -> list[Individual]:
        return [create_individual(max_depth) for _ in range(n_novelties)]

    return novelties


def create_lexicase(
    problem: Problem,
) -> Callable[[RandomSource, list[Individual], int], list[Individual]]:
    assert isinstance(problem, MultiObjectiveProblem)

    def lexicase(
        r: RandomSource,
        population: list[Individual],
        n_winners: int,
    ) -> list[Individual]:
        assert isinstance(problem, MultiObjectiveProblem)
        candidates = population.copy()
        # assert isinstance(candidates[0].fitness, list[float])
        n_cases = len(candidates[0].fitness)
        cases = r.shuffle(list(range(n_cases)))
        winners = []

        for _ in range(n_winners):
            candidates_to_check = candidates.copy()

            while len(candidates_to_check) > 1 and len(cases) > 0:
                new_candidates = list()
                c = cases[0]
                min_max_value = 0
                for i in range(len(candidates_to_check)):
                    checking_candidate = candidates_to_check[i]
                    check_value = checking_candidate.fitness[c]
                    if not new_candidates:
                        min_max_value = check_value
                        new_candidates.append(checking_candidate)
                    elif (check_value < min_max_value and problem.minimize[c]) or (
                        check_value > min_max_value and not problem.minimize[c]
                    ):
                        new_candidates.clear()
                        min_max_value = check_value
                        new_candidates.append(checking_candidate)
                    elif check_value == min_max_value:
                        new_candidates.append(checking_candidate)

                candidates_to_check = new_candidates.copy()
                cases.remove(c)

            winner = (
                r.choice(candidates_to_check)
                if len(candidates_to_check) > 1
                else candidates_to_check[0]
            )
            # assert isinstance(winner.fitness, list[float])
            winners.append(winner)
            candidates.remove(winner)
        return winners

    return lexicase
