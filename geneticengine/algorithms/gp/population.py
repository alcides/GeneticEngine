from typing import Iterator

from geneticengine.algorithms.api import ProgressTracker

from geneticengine.solutions import Individual


class Population:
    def __init__(self, it: Iterator[Individual], tracker: ProgressTracker):
        self.tracker = tracker
        self.individuals = []
        for ind in it:
            self.tracker.evaluate_single(ind)
            self.individuals.append(ind)

    def __iter__(self):
        return iter(self.individuals)

    def set_generation(self, gen: int):
        for ind in self.individuals:
            ind.metadata["generation"] = gen
