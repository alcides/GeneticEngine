from __future__ import annotations

import csv
from typing import Callable

from geneticengine.algorithms.gp.callback import Callback
from geneticengine.algorithms.gp.individual import Individual


class CSVCallback(Callback):
    """Callback that outputs to a given CSV file"""

    def __init__(
        self,
        filename: str = None,
        filter_population: Callable[[list[Individual]], list[Individual]] = lambda x: x,
    ):
        if filename is None:
            filename = "evolution_results.csv"
        self.filename = filename
        self.filter_population = filter_population
        self.cumulative_time = 0.0
        self.write_header()

    def end_evolution(self):
        self.outfile.close()

    def write_header(self):
        self.outfile = open(f"{self.filename}", "w", newline="")
        self.writer = csv.writer(self.outfile)
        self.writer.writerow(
            [
                "genotype_as_str",
                "fitness",
                "depth",
                "number_of_the_generation",
                "time_since_the_start_of_the_evolution",
                "seed",
            ],
        )

    def process_iteration(self, generation: int, population, time: float, gp):
        pop = self.filter_population(population)
        self.cumulative_time = self.cumulative_time + time
        for ind in pop:
            if hasattr(ind.genotype, "gengy_distance_to_term"):
                depth = ind.genotype.gengy_distance_to_term
            else:
                depth = -1
            row = [
                ind.genotype,
                ind.fitness,
                depth,
                generation,
                self.cumulative_time,
                gp.seed,
            ]
            self.writer.writerow([str(x) for x in row])
