from __future__ import annotations


from geneticengine.representations.api import RepresentationWithMutation


from geneticengine.algorithms.heuristics import HeuristicSearch
from geneticengine.solutions.individual import Individual, PhenotypicIndividual


class OnePlusOne(HeuristicSearch):
    """The (1 + 1) Evolutionary Algorithm."""

    def perform_search(self) -> list[Individual] | None:
        assert isinstance(self.representation, RepresentationWithMutation)
        current_ind = None
        while not self.is_done():
            if current_ind is None:
                n = self.representation.create_genotype(self.random)
                ind = PhenotypicIndividual(genotype=n, representation=self.representation)
            else:
                n2 = self.representation.mutate(self.random, ind.genotype)
                ind = PhenotypicIndividual(genotype=n2, representation=self.representation)
            self.tracker.evaluate([ind])
        return self.tracker.get_best_individuals()
