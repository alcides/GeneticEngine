from __future__ import annotations


from geneticengine.algorithms.heuristics import HeuristicSearch
from geneticengine.evaluation.budget import SearchBudget
from geneticengine.evaluation.recorder import SingleObjectiveProgressTracker
from geneticengine.problems import Problem
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import RepresentationWithMutation, SolutionRepresentation
from geneticengine.solutions.individual import Individual


class HC(HeuristicSearch):
    """Hill Climbing performs a local search within a neighbourhood."""

    def __init__(
        self,
        problem: Problem,
        budget: SearchBudget,
        representation: SolutionRepresentation,
        random: RandomSource = None,
        recorder: SingleObjectiveProgressTracker | None = None,
        number_of_mutations: int = 5,
    ):
        super().__init__(problem, budget, representation, random, recorder)
        self.number_of_mutations = number_of_mutations

    def search(self) -> Individual:
        assert isinstance(self.representation, RepresentationWithMutation)
        current_ind = None
        while not self.is_done():
            if current_ind is None:
                n = self.representation.instantiate(self.random)
                ind = Individual(genotype=n, genotype_to_phenotype=lambda x: self.representation.map(x))
                self.tracker.evaluate([ind])
            else:
                genotypes = [
                    self.representation.mutate(self.random, ind.genotype) for _ in range(self.number_of_mutations)
                ]
                neighbourhood = [
                    Individual(genotype=n2, genotype_to_phenotype=lambda x: self.representation.map(x))
                    for n2 in genotypes
                ]
                self.tracker.evaluate(neighbourhood)
            current_ind = self.tracker.get_best_individual()
        return self.tracker.get_best_individual()
