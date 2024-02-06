import csv
from time import time_ns
from typing import Any, Callable, Optional
from geneticengine.evaluation import Evaluator
from geneticengine.problems import Problem
from geneticengine.solutions import Individual


FieldMapper = Callable[[Individual, Problem], Any]


class SingleObjectiveProgressRecorder:
    evaluator: Evaluator
    best_individual: Optional[Individual]

    def __init__(
        self,
        evaluator: Evaluator,
        problem: Problem,
        csv_path: str = None,
        fields: dict[str, FieldMapper] = None,
        extra_fields: dict[str, FieldMapper] = None,
    ):
        self.start_time = time_ns()
        self.evaluator = evaluator
        self.problem = problem
        self.best_individual = None
        if csv_path is not None:
            self.csv_file = open(csv_path, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
        else:
            self.csv_writer = None
        if fields is not None:
            self.fields = fields
        else:
            self.fields = {
                "Execution Time": lambda i, _: time_ns() - self.start_time,
                "Phenotype": lambda i, _: i.get_phenotype(),
            }
            for comp in range(problem.number_of_objectives()):
                self.fields[f"Fitness{comp}"] = lambda i, p: i.get_fitness(p).fitness_components[comp]
            for name in extra_fields:
                self.fields[name] = extra_fields[name]

        self.csv_writer.writerow([name for name in self.fields])

    def eval(self, individuals: list[Individual]):
        for ind in self.evaluator.evaluate(self.problem, individuals):
            if self.best_individual is None:
                self.best_individual = ind
            elif self.problem.is_better(ind.get_fitness(self.problem), self.best_individual.get_fitness(self.problem)):
                self.best_individual = ind
            else:
                continue
            if self.csv_writer:
                self.csv_writer.writerow(
                    [self.fields[name](self.best_individual, self.problem) for name in self.fields],
                )
