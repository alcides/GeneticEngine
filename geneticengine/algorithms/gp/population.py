from typing import Iterator

from geneticengine.algorithms.api import ProgressTracker
from geneticengine.solutions.individual import PhenotypicIndividual


class Population:
    def __init__(self, it: Iterator[PhenotypicIndividual], tracker: ProgressTracker, generation: int = -1):
        self.tracker = tracker
        self.individuals = []
        for ind in it:
            ind.metadata["generation"] = generation
            self.tracker.evaluate_single(ind)
            self.individuals.append(ind)

    def __iter__(self):
        return iter(self.individuals)

    def get_individuals(self):
        return self.individuals
