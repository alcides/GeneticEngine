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
        all_cases = list(range(n_cases))

        assert isinstance(problem.minimize, list)

        for _ in range(target_size):
            candidates_to_check: list[PhenotypicIndividual] = candidates.copy()
            cases = random.shuffle(all_cases.copy())

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

class WeightedLexicaseSelection(LexicaseSelection):
    """
    Lexicase selection with accuracy-biased objective ordering.
    Uses weighted random permutation:
    - Objectives [0,1,2] (accuracy splits): weight 4.0 each
    - Objectives [3,4] (costs): weight 1.0 each
    """

    def __init__(self, epsilon: bool = True, objective_weights: list[float] | None = None):
        super().__init__(epsilon)
        self.objective_weights = objective_weights

    def _weighted_case_order(self, random:RandomSource, n_cases: int) -> list[int]:
        all_cases = list(range(n_cases))
        weights = self.objective_weights if self.objective_weights else [1.0] * n_cases

        if len(weights) != n_cases:
            return random.shuffle(all_cases.copy())

        clean = [float(w) if isinstance(w, (int, float)) and np.isfinite(w) and w > 0 else 0.0 for w in weights]
        if sum(clean) <= 0.0:
            return random.shuffle(all_cases.copy())

        remaining = all_cases.copy()
        order: list[int] = []

        while remaining:
            remaining_weights = [clean[i] for i in remaining]
            if sum(remaining_weights) <= 0.0:
                order.extend(random.shuffle(remaining.copy())) #prevent calling random.choice_weighted with all zero weights
                break
            chosen = random.choice_weighted(remaining, weights=remaining_weights) #pick from remaining cases with weighted probabilities
            order.append(chosen)
            remaining.remove(chosen)
        return order

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
        assert isinstance(problem.minimize, list)

        candidates = list(evaluator.evaluate(problem, list(population)))
        n_cases = problem.number_of_objectives()

        for _ in range(target_size):
            candidates_to_check = candidates.copy()
            cases = self._weighted_case_order(random, n_cases)

            while len(candidates_to_check) > 1 and cases:
                c = cases.pop(0)
                choose_best = min if problem.minimize[c] else max

                best_fitness = choose_best(
                    [x.get_fitness(problem).fitness_components[c] for x in candidates_to_check],
                )
                checking_value = best_fitness

                if self.epsilon:
                    vals = np.array(
                        [
                            x.get_fitness(problem).fitness_components[c]
                            for x in candidates_to_check
                            if not np.isnan(x.get_fitness(problem).fitness_components[c])
                        ],
                    )
                    mad = np.median(np.abs(vals - np.median(vals))) #mean absolute deviation from median
                    checking_value = best_fitness + mad if problem.minimize[c] else best_fitness - mad

                new_candidates = []
                for checking_candidate in candidates_to_check:
                    fitness = checking_candidate.get_fitness(problem).fitness_components[c]
                    ok = (fitness <= checking_value) if problem.minimize[c] else (fitness >= checking_value)
                    if ok:
                        new_candidates.append(checking_candidate)

                candidates_to_check = new_candidates

            winner = random.choice(candidates_to_check) if len(candidates_to_check) > 1 else candidates_to_check[0]
            yield winner
            candidates.remove(winner)

class PriorityLexicaseSelection(LexicaseSelection):
    """
    Priority Lexicase:
    - Lower integer means higher priority
    - Objectives in the same priority level are shuffled.
    Example: objective_priorities=[1, 1, 2, 3] -> [0/1 shuffled], then 2, then 3.
    """

    def __init__(self, epsilon: bool = True, objective_priorities: list[int] | None = None):
        super().__init__(epsilon)
        self.objective_priorities = objective_priorities

    def _priority_case_order(self, random:RandomSource, n_cases: int) -> list[int]:
        all_cases = list(range(n_cases))

        if self.objective_priorities is None:
            return random.shuffle(all_cases.copy()) #if no priorities given, just random order

        if len(self.objective_priorities) != n_cases:
            raise ValueError("Length of objective_priorities must match number of objectives")

        levels: dict[int, list[int]] = {}
        for i, p in enumerate(self.objective_priorities):
            if not isinstance(p, int):
                raise ValueError(f"Invalid priority {p} for objective {i}.")
            levels.setdefault(p, []).append(i)

        order: list[int] = []
        for level in sorted(levels):
            order.extend(random.shuffle(levels[level].copy())) #random order within same priority level

        return order

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
        assert isinstance(problem.minimize, list)

        candidates = list(evaluator.evaluate(problem, list(population)))
        n_cases = problem.number_of_objectives()

        for _ in range(target_size):
            candidates_to_check = candidates.copy()
            cases = self._priority_case_order(random, n_cases)

            while len(candidates_to_check) > 1 and cases:
                new_candidates: list[PhenotypicIndividual] = []
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
                    fitness = checking_candidate.get_fitness(problem).fitness_components[c]
                    ok = (fitness <= checking_value) if problem.minimize[c] else (fitness >= checking_value)
                    if ok:
                        new_candidates.append(checking_candidate)
                candidates_to_check = new_candidates
            winner = random.choice(candidates_to_check) if len(candidates_to_check) > 1 else candidates_to_check[0]
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
