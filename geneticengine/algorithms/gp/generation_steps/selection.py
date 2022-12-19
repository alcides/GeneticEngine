from __future__ import annotations

from copy import deepcopy
from typing import Callable
from typing import List
from typing import Union
import numpy as np

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
    epsilon: bool = False,
) -> Callable[[RandomSource, list[Individual], int], list[Individual]]:
    """
    The create_lexicase is a function that uses the lexicase selection algorithm to select a list of
    Individuals with the best fitness

    Args:
        problem: type of problem that you are trying to solve
        epsilon: if True, espilon-lexicase is performed. We use the method given by equation 5 in https://dl.acm.org/doi/pdf/10.1145/2908812.2908898.

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
                best_fitness = min(list(map(lambda x: x.fitness[c], candidates_to_check))) if problem.minimize[c] else max(list(map(lambda x: x.fitness[c], candidates_to_check))) # type: ignore
                checking_value = best_fitness
                if epsilon:
                    fitness_values = [ x for x in  map(lambda x: x.fitness[c], candidates_to_check) if not np.isnan(x) ] # type: ignore
                    mad = np.median(np.absolute(fitness_values - np.median(fitness_values)))
                    checking_value = best_fitness + mad if problem.minimize[c] else best_fitness - mad
                for i in range(len(candidates_to_check)):
                    checking_candidate = candidates_to_check[i]
                    add_candidate = checking_candidate.fitness[c] <= checking_value if problem.minimize[c] else checking_candidate.fitness[c] >= checking_value # type: ignore
                    if add_candidate:
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
