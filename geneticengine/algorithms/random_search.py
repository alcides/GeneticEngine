from __future__ import annotations

from geneticengine.algorithms.heuristics import HeuristicSearch

from geneticengine.solutions.individual import Individual


class RandomSearch(HeuristicSearch):
    """Randomly generates new solutions and keeps the best one."""

    def search(self) -> Individual:
        while not self.is_done():
            n = self.representation.instantiate(self.random)
            ind = Individual(genotype=n, genotype_to_phenotype=lambda x: self.representation.map(x))
            self.tracker.evaluate([ind])
        return self.tracker.get_best_individual()
