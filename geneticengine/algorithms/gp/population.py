from typing import Iterator

from geneticengine.algorithms.api import ProgressTracker
from geneticengine.solutions.individual import PhenotypicIndividual


class Population:
    def __init__(self, it: Iterator[PhenotypicIndividual], tracker: ProgressTracker, generation: int = -1):
        self.tracker = tracker
        self.individuals = []
        individual_list = list(it)
        for ind in individual_list:
            ind.metadata["generation"] = generation
            self.individuals.append(ind)
        self.tracker.evaluate(individual_list)

    def __iter__(self):
        return iter(self.individuals)

    def get_individuals(self):
        return self.individuals
