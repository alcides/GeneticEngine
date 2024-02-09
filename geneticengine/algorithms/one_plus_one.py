from __future__ import annotations


from geneticengine.representations.api import RepresentationWithMutation


from geneticengine.algorithms.heuristics import HeuristicSearch
from geneticengine.solutions.individual import Individual


class OnePlusOne(HeuristicSearch):
    """The (1 + 1) Evolutionary Algorithm"""

    def search(self) -> Individual:
        assert isinstance(self.representation, RepresentationWithMutation)
        current_ind = None
        while not self.is_done():
            if current_ind is None:
                n = self.representation.instantiate(self.random)
                ind = Individual(genotype=n, genotype_to_phenotype=lambda x: self.representation.map(x))
            else:
                n2 = self.representation.mutate(self.random, ind.genotype)
                ind = Individual(genotype=n2, genotype_to_phenotype=lambda x: self.representation.map(x))
            self.tracker.evaluate([ind])
        return self.tracker.get_best_individual()
