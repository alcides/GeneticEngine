from __future__ import annotations

from copy import deepcopy
from typing import Callable
from typing import List
from typing import Union

from geneticengine.algorithms.gp.individual import Individual
from geneticengine.core.problems import MultiObjectiveProblem
from geneticengine.core.problems import Problem
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.random.sources import RandomSource


def create_tournament(
    tournament_size: int,
    problem: Problem,
) -> Callable[[RandomSource, list[Individual], int], list[Individual]]:
    """
    The create_tournament is a function that uses the tournament selection algorithm to select a list of individuals with the best fitness

    Args:
        tournament_size: number of individuals from the population that will be randomly selected
        problem: type of problem that you are trying to solve

    Returns:
        A callable object that returns a list of winners
    """
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
) -> Callable[
    [
        list[Individual],
        Problem,
        Callable[[Problem, list[Individual]], Individual],
        Callable[[Individual], float | list[float]],
    ],
    list[Individual],
]:
    """
    The create_elitism is a function that returns the individuals with the best fitness in a generation

    Args:
        n_elites: number of desired  elite Individuals
    Returns:
        A callable object that returns a list of elite Individuals
    """

    def elitism(
        population: list[Individual],
        problem: Problem,
        best_individual_function: Callable[[Problem, list[Individual]], Individual],
        evaluate: Callable[[Individual], float | list[float]],
    ) -> list[Individual]:
        fitnesses = [evaluate(x) for x in population]

        if isinstance(problem, SingleObjectiveProblem):
            assert all(isinstance(x, float) for x in fitnesses)
        else:
            assert all(isinstance(x, list) for x in fitnesses)

        elites: list[Individual] = list()
        population_copy = population.copy()
        i = 0
        while len(elites) < n_elites and i < len(population):

            elite = best_individual_function(problem, population_copy)
            elites.append(elite)
            population_copy.remove(elite)
            i += 1

        return elites

    return elitism


def create_novelties(
    create_individual: Callable[[int], Individual],
    max_depth: int,
) -> Callable[[int], list[Individual]]:
    """
    The create_novelties is a function that returns a list of completely new Individuals

    Args:
        create_individual: callable object that returns a single Individual
        max_depth:
    Returns:
        A callable object that returns a list of new Individuals
    """

    def novelties(n_novelties: int) -> list[Individual]:
        return [create_individual(max_depth) for _ in range(n_novelties)]

    return novelties


def create_lexicase(
    problem: Problem,
) -> Callable[[RandomSource, list[Individual], int], list[Individual]]:
    """
    The create_lexicase is a function that uses the lexicase selection algorithm to select a list of
    Individuals with the best fitness

    Args:
        problem: type of problem that you are trying to solve

    Returns:
        A callable object that returns a list of the selected Individuals
    """
    assert isinstance(problem, MultiObjectiveProblem)

    def lexicase(
        r: RandomSource,
        population: list[Individual],
        n_winners: int,
    ) -> list[Individual]:
        assert isinstance(problem, MultiObjectiveProblem)
        candidates = population.copy()
        assert isinstance(candidates[0].fitness, list)
        n_cases = len(candidates[0].fitness)
        cases = r.shuffle(list(range(n_cases)))
        winners = []

        for _ in range(n_winners):
            candidates_to_check = candidates.copy()

            while len(candidates_to_check) > 1 and len(cases) > 0:
                new_candidates: list[Individual] = list()
                c = cases[0]
                min_max_value = 0
                for i in range(len(candidates_to_check)):
                    checking_candidate = candidates_to_check[i]
                    assert isinstance(checking_candidate.fitness, list)
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
            assert isinstance(winner.fitness, list)
            winners.append(winner)
            candidates.remove(winner)
        return winners

    return lexicase
