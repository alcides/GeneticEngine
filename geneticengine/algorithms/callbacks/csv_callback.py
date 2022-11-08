from __future__ import annotations

import csv
from typing import Callable
from typing import Optional

from geneticengine.algorithms.callbacks.callback import Callback
from geneticengine.algorithms.gp.individual import Individual


class CSVCallback(Callback):
    """Callback that outputs to a given CSV file."""

    def __init__(
        self,
        filename: str | None = None,
        filter_population: Callable[[list[Individual]], list[Individual]] = lambda x: x,
        test_data: Callable[[Individual], float] | None = None,
        only_record_best_ind: bool = True,
        save_genotype_as_string: bool = True,
    ):
        if filename is None:
            filename = "evolution_results.csv"
        self.filename = filename
        self.filter_population = filter_population
        self.time = 0.0
        self.test_data = test_data
        self.only_record_best_ind = only_record_best_ind
        self.save_genotype_as_string = save_genotype_as_string
        self.write_header()

    def end_evolution(self):
        self.outfile.close()

    def write_header(self):
        self.outfile = open(f"{self.filename}", "w", newline="")
        self.writer = csv.writer(self.outfile)
        row = [
            "fitness",
            "depth",
            "nodes",
            "number_of_the_generation",
            "time_since_the_start_of_the_evolution",
            "seed",
        ]
        if self.test_data:
            row.append("test_fitness")
        if self.save_genotype_as_string:
            row.append("genotype_as_str")
        self.writer.writerow(row)

    def process_iteration(self, generation: int, population, time: float, gp):
        pop = self.filter_population(population)
        if self.only_record_best_ind:
            pop = [gp.get_best_individual(gp.problem, population)]
        self.time = time
        for ind in pop:
            if hasattr(ind.genotype, "gengy_distance_to_term"):
                depth = ind.genotype.gengy_distance_to_term
            else:
                depth = -1
            if hasattr(ind.genotype, "gengy_nodes"):
                nodes = ind.genotype.gengy_nodes
            else:
                nodes = -1
            row = [
                ind.fitness,
                depth,
                nodes,
                generation,
                self.time,
                gp.seed,
            ]
            if self.test_data:
                row.append(self.test_data(ind))
            if self.save_genotype_as_string:
                row.append(ind.genotype)
            self.writer.writerow([str(x) for x in row])
