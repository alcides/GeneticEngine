from __future__ import annotations
from typing import Iterator
import numpy as np

from geneticengine.solutions.individual import PhenotypicIndividual
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.problems import Fitness, MultiObjectiveProblem
from geneticengine.problems import Problem
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import Representation
from geneticengine.evaluation import Evaluator


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
        random: RandomSource,
        population: Iterator[PhenotypicIndividual],
        target_size: int,
        generation: int,
    ) -> Iterator[PhenotypicIndividual]:
        initial = list(population)
        candidates : list[PhenotypicIndividual] = list(evaluator.evaluate(problem, initial))
        if not candidates:
            yield from initial

        if problem.number_of_objectives() > 1:
            goal = random.randint(0, problem.number_of_objectives()-1)
        else:
            goal = 0

        for _ in range(target_size):
            candidates = [random.choice(candidates) for _ in range(self.tournament_size)]

            winner = max(candidates, key=lambda ind: ind.get_fitness(problem).fitness_components[goal])
            yield winner

            if not self.with_replacement:
                candidates.remove(winner)
                if not candidates:
                    candidates = list(population)


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
        random: RandomSource,
        population: Iterator[PhenotypicIndividual],
        target_size: int,
        generation: int,
    ) -> Iterator[PhenotypicIndividual]:
        assert isinstance(problem, MultiObjectiveProblem)
        candidates = list(evaluator.evaluate(problem, list(population)))
        n_cases = problem.number_of_objectives()
        cases = random.shuffle(list(range(n_cases)))

        assert isinstance(problem.minimize, list)

        for _ in range(target_size):
            candidates_to_check: list[PhenotypicIndividual] = candidates.copy()

            while len(candidates_to_check) > 1 and cases:
                new_candidates: list[PhenotypicIndividual] = list()
                c = cases.pop(0)

                choose_best = min if problem.minimize[c] else max

                best_fitness = choose_best([x.get_fitness(problem).fitness_components[c] for x in candidates_to_check])
                checking_value = best_fitness

                if self.epsilon:

                    def get_fitness_value(ind: PhenotypicIndividual, c: int):
                        fit = ind.get_fitness(problem)
                        return fit.fitness_components[c]

                    fitness_values = np.array(
                        [get_fitness_value(x, c) for x in candidates_to_check if not np.isnan(get_fitness_value(x, c))],
                    )
                    mad = np.median(np.absolute(fitness_values - np.median(fitness_values)))
                    checking_value = best_fitness + mad if problem.minimize[c] else best_fitness - mad

                for checking_candidate in candidates_to_check:
                    fitness: Fitness = checking_candidate.get_fitness(problem)
                    if problem.minimize[c]:
                        add_candidate = fitness.fitness_components[c] <= checking_value
                    else:
                        add_candidate = fitness.fitness_components[c] >= checking_value
                    if add_candidate:
                        new_candidates.append(checking_candidate)

                candidates_to_check = new_candidates.copy()

            winner = random.choice(candidates_to_check) if len(candidates_to_check) > 1 else candidates_to_check[0]
            assert isinstance(winner.get_fitness(problem).fitness_components, list)
            yield winner
            candidates.remove(winner)





class InformedDownsamplingSelection(GeneticStep):
    """
    Selects individuals using only test cases with highest variance.
    Faster than standard Lexicase by reducing test case evaluations.
    """

    def __init__(self, max_sample_size: int = 10, percent: float = 0.1):
        self.max_sample_size = max_sample_size
        self.percent = percent

    def iterate(
        self,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random: RandomSource,
        population: Iterator[PhenotypicIndividual],
        target_size: int,
        generation: int,
    ) -> Iterator[PhenotypicIndividual]:
        assert isinstance(problem, MultiObjectiveProblem)

        candidates = list(evaluator.evaluate(problem, list(population)))
        n_cases = problem.number_of_objectives()
        n_candidates = len(candidates)

        fitness_matrix = np.array([
            ind.get_fitness(problem).fitness_components
            for ind in candidates
        ])

        case_variances = [
            (i, np.var(fitness_matrix[:, i], ddof=1)) for i in range(n_cases)
        ]

        sample_size = min(self.max_sample_size, max(1, int(self.percent * n_cases)))
        sample_size = min(sample_size, n_cases)
        case_variances.sort(key=lambda x: x[1], reverse=True)
        selected_cases = [i for i, _ in case_variances[:sample_size]]

        assert isinstance(problem.minimize, list)

        selected_indices = list(range(n_candidates))
        for _ in range(target_size):
            pool_indices = selected_indices.copy()

            for c in selected_cases:
                scores = fitness_matrix[pool_indices, c]
                choose_best = np.min if problem.minimize[c] else np.max
                best_score = choose_best(scores)
                pool_indices = [
                    i for i in pool_indices if fitness_matrix[i, c] == best_score
                ]
                if len(pool_indices) <= 1:
                    break

            winner_idx = pool_indices[0] if pool_indices else random.randint(0, len(selected_indices) - 1)
            winner = candidates[winner_idx] if pool_indices else candidates[winner_idx]

            yield winner
            selected_indices.remove(winner_idx)
