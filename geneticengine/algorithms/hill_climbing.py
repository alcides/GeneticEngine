from __future__ import annotations


from geneticengine.algorithms.heuristics import HeuristicSearch
from geneticengine.evaluation.budget import SearchBudget
from geneticengine.evaluation.tracker import ProgressTracker
from geneticengine.problems import Problem
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import RepresentationWithMutation, Representation
from geneticengine.solutions.individual import Individual, PhenotypicIndividual


class HC(HeuristicSearch):
    """Hill Climbing performs a local search within a neighbourhood."""

    def __init__(
        self,
        problem: Problem,
        budget: SearchBudget,
        representation: Representation,
        random: RandomSource = None,
        tracker: ProgressTracker | None = None,
        number_of_mutations: int = 5,
    ):
        super().__init__(problem, budget, representation, random, tracker)
        self.number_of_mutations = number_of_mutations

    def perform_search(self) -> list[Individual] | None:
        assert isinstance(self.representation, RepresentationWithMutation)
        current_ind = None
        while not self.is_done():
            if current_ind is None:
                n = self.representation.create_genotype(self.random)
                ind = PhenotypicIndividual(genotype=n, representation=self.representation)
                self.tracker.evaluate([ind])
            else:
                genotypes = [
                    self.representation.mutate(self.random, ind.genotype) for _ in range(self.number_of_mutations)
                ]
                neighbourhood = [
                    PhenotypicIndividual(genotype=n2, representation=self.representation) for n2 in genotypes
                ]
                self.tracker.evaluate(neighbourhood)
            current_ind = self.tracker.get_best_individuals()[0]
        return self.tracker.get_best_individuals()
