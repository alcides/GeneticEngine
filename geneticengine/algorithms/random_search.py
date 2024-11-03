from __future__ import annotations

from geneticengine.algorithms.heuristics import HeuristicSearch
from geneticengine.solutions.individual import Individual, PhenotypicIndividual


class RandomSearch(HeuristicSearch):
    """Randomly generates new solutions and keeps the best one."""

    def perform_search(self) -> list[Individual] | None:
        while not self.is_done():
            n = self.representation.create_genotype(self.random)
            ind = PhenotypicIndividual(genotype=n, representation=self.representation)
            self.tracker.evaluate([ind])
        return self.tracker.get_best_individuals()
