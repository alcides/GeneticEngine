from __future__ import annotations

import numpy as np

from geneticengine.algorithms.gp.individual import Individual
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.core.problems import Fitness, MultiObjectiveProblem
from geneticengine.core.problems import Problem
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import Representation
from geneticengine.core.evaluators import Evaluator


class TournamentSelection(GeneticStep):
    """TournamentSelection represents a tournament selection algorithm, where
    tournament_size individuals are selected at random, and only the best
    passes to the next generation."""

    def __init__(self, tournament_size: int, with_replacement: bool = False):
        """
        Args:
            tournament_size (int): number of individuals from the population that will be randomly selected
            with_replacement (bool): whether the selected individuals can appear again in another tournament (default: False)
        """
        self.tournament_size = tournament_size
        self.with_replacement = with_replacement

    def iterate(
        self,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random_source: Source,
        population: list[Individual],
        target_size: int,
        generation: int,
    ) -> list[Individual]:
        assert isinstance(problem, SingleObjectiveProblem)
        winners: list[Individual] = []
        candidates = population.copy()
        evaluator.eval(problem, candidates)
        for _ in range(target_size):
            candidates = [random_source.choice(population) for _ in range(self.tournament_size)]
            winner = max(candidates, key=Individual.key_function(problem))
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
            epsilon: if True, espilon-lexicase is performed. We use the method given by equation 5 in
                https://dl.acm.org/doi/pdf/10.1145/2908812.2908898.
        """
        self.epsilon = epsilon

    def iterate(
        self,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random_source: Source,
        population: list[Individual],
        target_size: int,
        generation: int,
    ) -> list[Individual]:
        assert isinstance(problem, MultiObjectiveProblem)
        candidates = population.copy()
        evaluator.eval(problem, candidates)
        n_cases = problem.number_of_objectives()
        cases = random_source.shuffle(list(range(n_cases)))
        winners = []
        minimize : list[bool]

        if n_cases == 1 and isinstance(problem.minimize, bool):
            n_cases = len(candidates[0].get_fitness(problem).fitness_components)
            minimize = [problem.minimize for _ in range(n_cases)]
        else:
            assert isinstance(problem.minimize, list)
            minimize = problem.minimize

        assert n_cases == len(candidates[0].get_fitness(problem).fitness_components)

        for _ in range(target_size):
            candidates_to_check = candidates.copy()

            while len(candidates_to_check) > 1 and cases:
                new_candidates: list[Individual] = list()
                c = cases.pop(0)

                choose_best = min if minimize[c] else max

                best_fitness = choose_best([x.get_fitness(problem).fitness_components[c] for x in candidates_to_check])
                checking_value = best_fitness

                if self.epsilon:

                    def get_fitness_value(ind: Individual, c: int):
                        (summary, values) = ind.get_fitness(problem)
                        return values[c]

                    fitness_values = np.array(
                        [get_fitness_value(x, c) for x in candidates_to_check if not np.isnan(get_fitness_value(x, c))],
                    )
                    mad = np.median(np.absolute(fitness_values - np.median(fitness_values)))
                    checking_value = best_fitness + mad if minimize[c] else best_fitness - mad

                for checking_candidate in candidates_to_check:
                    fitness: Fitness = checking_candidate.get_fitness(problem)
                    if minimize[c]:
                        add_candidate = fitness.fitness_components[c] <= checking_value
                    else:
                        add_candidate = fitness.fitness_components[c] >= checking_value
                    if add_candidate:
                        new_candidates.append(checking_candidate)

                candidates_to_check = new_candidates.copy()

            winner = (
                random_source.choice(candidates_to_check) if len(candidates_to_check) > 1 else candidates_to_check[0]
            )
            assert isinstance(winner.get_fitness(problem).fitness_components, list)
            winners.append(winner)
            candidates.remove(winner)
        return winners
