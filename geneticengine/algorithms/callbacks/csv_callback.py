from __future__ import annotations

import csv
from typing import Callable

from geneticengine.algorithms.callbacks.callback import Callback
from geneticengine.algorithms.gp.individual import Individual


class CSVCallback(Callback):
    """Callback that outputs to a given CSV file"""

    def __init__(
        self,
        filename: str = None,
        filter_population: Callable[[list[Individual]], list[Individual]] = lambda x: x,
        save_genotype_as_string: bool = True,
    ):
        if filename is None:
            filename = "evolution_results.csv"
        self.filename = filename
        self.filter_population = filter_population
        self.cumulative_time = 0.0
        self.save_genotype_as_string = save_genotype_as_string
        self.write_header()

    def end_evolution(self):
        self.outfile.close()

    def write_header(self):
        self.outfile = open(f"{self.filename}", "w", newline="")
        self.writer = csv.writer(self.outfile)
        if self.save_genotype_as_string:
            self.writer.writerow(
                [
                    "fitness",
                    "depth",
                    "nodes",
                    "number_of_the_generation",
                    "time_since_the_start_of_the_evolution",
                    "seed",
                    "genotype_as_str",
                ],
            )
        else:
            self.writer.writerow(
                [
                    "fitness",
                    "depth",
                    "nodes",
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
            if hasattr(ind.genotype, "gengy_nodes"):
                nodes = ind.genotype.gengy_nodes
            else:
                nodes = -1
            if self.save_genotype_as_string:
                row = [
                    ind.fitness,
                    depth,
                    nodes,
                    generation,
                    self.cumulative_time,
                    gp.seed,
                    ind.genotype,
                ]
            else:
                row = [
                    ind.fitness,
                    depth,
                    nodes,
                    generation,
                    self.cumulative_time,
                    gp.seed,
                ]
            self.writer.writerow([str(x) for x in row])
