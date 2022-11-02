from __future__ import annotations

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
    ) -> list[Individual]:
        assert isinstance(problem, SingleObjectiveProblem)
        winners: list[Individual] = []
        candidates = population.copy()
        for _ in range(target_size):
            candidates = [r.choice(population) for _ in range(self.tournament_size)]
            if problem.minimize:
                winner = min(candidates, key=lambda x: x.evaluate(problem))
            else:
                winner = max(candidates, key=lambda x: x.evaluate(problem))
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

    def iterate(
        self,
        problem: Problem,
        representation: Representation,
        r: Source,
        population: list[Individual],
        target_size: int,
    ) -> list[Individual]:
        assert isinstance(problem, MultiObjectiveProblem)
        candidates = population.copy()
        assert isinstance(candidates[0].fitness, list)
        n_cases = len(candidates[0].fitness)
        cases = r.shuffle(list(range(n_cases)))
        winners = []

        for _ in range(target_size):
            candidates_to_check = candidates.copy()

            while len(candidates_to_check) > 1 and len(cases) > 0:
                new_candidates: list[Individual] = list()
                for candidate in new_candidates:
                    candidate.evaluate(problem)
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
