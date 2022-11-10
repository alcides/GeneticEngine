from __future__ import annotations

import csv
from typing import Any, Callable
from typing import Optional

from geneticengine.algorithms.callbacks.callback import Callback
from geneticengine.algorithms.gp.individual import Individual


class CSVCallback(Callback):
    """
    Callback that outputs to a given CSV file.
    
    Args:
        filename (str | None): The file name where the results are saved to (default: evolution_results.csv).
        filter_population (Callable[[list[Individual]], list[Individual]]): The method of filtering the population before saving (default: identity function, aka: lambda x: x).
        test_data (Callable[[Any], float] | None): A fitness_function-style function to save test results (genotype -> test_fitness).
        only_record_best_ind (bool): Set to False if you want to save data on all individuals. This is memory expensive (default: True).
        save_genotype_as_string (bool): Set to True if you want to save the genotype if the individuals. This is memory expensive, especially in combination with the only_record_best_ind set to False (default: False).
        save_productions (bool): Set to True to save the occurences of productions of individuals (default: False).

    """

    def __init__(
        self,
        filename: str | None = None,
        filter_population: Callable[[list[Individual]], list[Individual]] = lambda x: x,
        test_data: Callable[[Any], float] | None = None,
        only_record_best_ind: bool = True,
        save_genotype_as_string: bool = False,
        save_productions: bool = False,
    ):
        if filename is None:
            filename = "evolution_results.csv"
        self.filename = filename
        self.filter_population = filter_population
        self.time = 0.0
        self.test_data = test_data
        self.only_record_best_ind = only_record_best_ind
        self.save_genotype_as_string = save_genotype_as_string
        self.save_productions = save_productions
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
            "time_since_the_start_of_the_evolution",
            "seed",
        ]
        if self.test_data:
            row.append("Test fitness")
        if self.save_genotype_as_string:
            row.append("genotype_as_str")
        if self.save_productions:
            row.append("productions")
        self.writer.writerow(row)

    def process_iteration(self, generation: int, population, time: float, gp):
        pop = self.filter_population(population)
        if self.only_record_best_ind:
            pop = [gp.get_best_individual(gp.problem, population)]
        self.time = time
        for ind in pop:
            phenotype = gp.representation.genotype_to_phenotype(
                        gp.grammar,
                        ind.genotype,
                    )
            if hasattr(phenotype, "gengy_distance_to_term"):
                depth = phenotype.gengy_distance_to_term
            else:
                depth = -1
            if hasattr(phenotype, "gengy_nodes"):
                nodes = phenotype.gengy_nodes
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
                row.append(self.test_data(phenotype))
            if self.save_genotype_as_string:
                row.append(ind.genotype)
            if self.save_productions:
                row.append(ind.count_prods(gp.representation.genotype_to_phenotype, gp.grammar))
            self.writer.writerow([str(x) for x in row])
