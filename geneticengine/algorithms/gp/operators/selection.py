from __future__ import annotations

import numpy as np

from geneticengine.algorithms.gp.individual import Individual
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.core.problems import MultiObjectiveProblem
from geneticengine.core.problems import Problem
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import Representation


class TournamentSelection(GeneticStep):
    """TournamentSelection represents a tournament selection algorithm, where
    tournament_size individuals are selected at random, and only the best
    passes to the next generation."""

    def __init__(self, tournament_size: int, with_replacement: bool = False):
        """
        Args:
            tournament_size (int): number of individuals from the population that will be randomly selected
        """
        self.tournament_size = tournament_size
        self.with_replacement = with_replacement

    def iterate(
        self,
        problem: Problem,
        representation: Representation,
        r: Source,
        population: list[Individual],
        target_size: int,
        generation: int,
    ) -> list[Individual]:
        assert isinstance(problem, SingleObjectiveProblem)
        winners: list[Individual] = []
        candidates = population.copy()
        for _ in range(target_size):
            candidates = [r.choice(population) for _ in range(self.tournament_size)]
            if problem.minimize:
                winner = min(
                    candidates,
                    key=lambda x: x.evaluate(
                        problem,
                    ),
                )
            else:
                winner = max(
                    candidates,
                    key=lambda x: x.evaluate(
                        problem,
                    ),
                )
            winners.append(winner)
            if self.with_replacement:
                candidates.remove(winner)
                if not candidates:
                    candidates = population.copy()
        assert len(winners) == target_size
        return winners


class LexicaseSelection(GeneticStep):
    """Implements Lexicase Selection
    (http://williamlacava.com/research/lexicase/)."""

    def __init__(self, epsilon: bool = False):
        """
        Args:
         epsilon: if True, espilon-lexicase is performed. We use the method given by equation 5 in https://dl.acm.org/doi/pdf/10.1145/2908812.2908898.
        """
        self.epsilon = epsilon

    def iterate(
        self,
        problem: Problem,
        representation: Representation,
        r: Source,
        population: list[Individual],
        target_size: int,
        generation: int,
    ) -> list[Individual]:
        assert isinstance(problem, MultiObjectiveProblem)
        candidates = population.copy()
        candidates[0].evaluate(problem)
        assert isinstance(candidates[0].fitness, list)
        n_cases = len(candidates[0].fitness)
        cases = r.shuffle(list(range(n_cases)))
        winners = []

        for cand in candidates:
            cand.evaluate(problem=problem)

        for _ in range(target_size):
            candidates_to_check = candidates.copy()

            while len(candidates_to_check) > 1 and cases:
                new_candidates: list[Individual] = list()
                c = cases.pop(0)

                choose_best = min if problem.minimize[c] else max

                best_fitness = choose_best([x.fitness[c] for x in candidates_to_check])  # type: ignore

                checking_value = best_fitness
                if self.epsilon:
                    fitness_values = np.array([x.fitness[c] for x in candidates_to_check if not np.isnan(x.fitness[c])])  # type: ignore
                    mad = np.median(np.absolute(fitness_values - np.median(fitness_values)))
                    checking_value = best_fitness + mad if problem.minimize[c] else best_fitness - mad

                for checking_candidate in candidates_to_check:
                    add_candidate = checking_candidate.fitness[c] <= checking_value if problem.minimize[c] else checking_candidate.fitness[c] >= checking_value  # type: ignore
                    if add_candidate:
                        new_candidates.append(checking_candidate)

                candidates_to_check = new_candidates.copy()

            winner = r.choice(candidates_to_check) if len(candidates_to_check) > 1 else candidates_to_check[0]
            assert isinstance(winner.fitness, list)
            winners.append(winner)
            candidates.remove(winner)
        return winners
