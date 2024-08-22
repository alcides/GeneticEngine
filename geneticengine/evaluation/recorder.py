from abc import ABC
import csv
from time import monotonic_ns
from typing import Any, Callable
from geneticengine.problems import Problem
from geneticengine.solutions import Individual

FieldMapper = Callable[[Any, Individual, Problem], Any]


class SearchRecorder(ABC):
    def register(self, tracker: Any, individual: Individual, problem: Problem, is_best: bool): ...


class CSVSearchRecorder(SearchRecorder):
    def __init__(
        self,
        csv_path: str,
        problem: Problem,
        fields: dict[str, FieldMapper] | None = None,
        extra_fields: dict[str, FieldMapper] | None = None,
        only_record_best_individuals: bool = True,
    ):
        assert csv_path is not None
        self.csv_file = open(csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        if fields is not None:
            self.fields = fields
        else:
            self.fields = {
                "Execution Time": lambda t, i, _: (monotonic_ns() - t.start_time) * 0.000000001,
                "Phenotype": lambda t, i, _: i.get_phenotype(),
            }
            for comp in range(problem.number_of_objectives()):
                self.fields[f"Fitness{comp}"] = lambda t, i, p: i.get_fitness(p).fitness_components[comp]
        if extra_fields is not None:
            for name in extra_fields:
                self.fields[name] = extra_fields[name]
        self.csv_writer.writerow([name for name in self.fields])
        self.csv_file.flush()
        self.header_printed = False
        self.only_record_best_individuals = only_record_best_individuals

    def register(self, tracker: Any, individual: Individual, problem: Problem, is_best=False):
        if not self.only_record_best_individuals or is_best:
            self.csv_writer.writerow(
                [self.fields[name](tracker, individual, problem) for name in self.fields],
            )
            self.csv_file.flush()
