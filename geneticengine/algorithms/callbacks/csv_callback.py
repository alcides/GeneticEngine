from __future__ import annotations

import csv
from typing import Any
from typing import Callable

from geneticengine.algorithms.callbacks.callback import Callback
from geneticengine.algorithms.gp.individual import Individual


class CSVCallback(Callback):
    """Callback that outputs to a given CSV file."""

    def __init__(
        self,
        filename: str = "evolution_results.csv",
        filter_population: Callable[[list[Individual]], list[Individual]] = lambda x: x,
        only_record_best_ind: bool = True,
        extra_columns: dict[
            str,
            Callable[[int, list[Individual], float, Any, Individual], Any],
        ]
        | None = None,
    ):
        self.filename = filename
        self.filter_population = filter_population
        self.time = 0.0
        self.only_record_best_ind = only_record_best_ind
        self.extra_columns: dict[str, Callable[[int, list[Individual], float, Any, Individual], Any]] = (
            extra_columns if extra_columns else {}
        )
        self.write_header()

    def end_evolution(self):
        self.outfile.close()

    def write_header(self):
        self.outfile = open(f"{self.filename}", "w", newline="")
        self.writer = csv.writer(self.outfile)
        row = [
            "Fitness",
            "Depth",
            "Nodes",
            "Generations",
            "Execution Time",
            "Seed",
        ]
        for name in self.extra_columns:
            row.append(name)
        self.writer.writerow(row)

    def process_iteration(self, generation: int, population, time: float, gp):
        pop = self.filter_population(population)
        if self.only_record_best_ind:
            pop = [gp.get_best_individual(gp.problem, population)]
        self.time = time
        for ind in pop:
            phenotype = ind.get_phenotype()
            if hasattr(phenotype, "gengy_distance_to_term"):
                depth = phenotype.gengy_distance_to_term
            else:
                depth = -1
            if hasattr(phenotype, "gengy_nodes"):
                nodes = phenotype.gengy_nodes
            else:
                nodes = -1
            row = [ind.get_fitness(gp.problem), depth, nodes, phenotype, generation, self.time, gp.random_source.seed]

            for name, fun in self.extra_columns.items():
                row.append(fun(generation, population, time, gp, ind))

            self.writer.writerow([str(x) for x in row])
