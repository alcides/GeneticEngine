import csv
from time import time_ns
from typing import Any, Callable, Optional
from geneticengine.evaluation import Evaluator
from geneticengine.evaluation.sequential import SequentialEvaluator
from geneticengine.problems import Problem
from geneticengine.solutions import Individual


FieldMapper = Callable[[Individual, Problem], Any]


class SingleObjectiveProgressTracker:
    evaluator: Evaluator
    best_individual: Optional[Individual]

    def __init__(
        self,
        problem: Problem,
        evaluator: Evaluator = None,
        csv_path: str = None,
        fields: dict[str, FieldMapper] = None,
        extra_fields: dict[str, FieldMapper] = None,
    ):
        self.start_time = time_ns()
        if evaluator is None:
            self.evaluator = SequentialEvaluator()
        else:
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
            for comp in range(self.problem.number_of_objectives()):
                self.fields[f"Fitness{comp}"] = lambda i, p: i.get_fitness(p).fitness_components[comp]
            if extra_fields is not None:
                for name in extra_fields:
                    self.fields[name] = extra_fields[name]
        if self.csv_writer:
            self.csv_writer.writerow([name for name in self.fields])

    def evaluate(self, individuals: list[Individual]):
        problem = self.problem
        self.evaluator.evaluate(problem, individuals)
        for ind in individuals:
            if self.best_individual is None:
                self.best_individual = ind
            elif problem.is_better(ind.get_fitness(problem), self.best_individual.get_fitness(problem)):
                self.best_individual = ind
            else:
                continue
            if self.csv_writer:
                self.csv_writer.writerow(
                    [self.fields[name](self.best_individual, problem) for name in self.fields],
                )

    def get_best_individual(self) -> Individual:
        return self.best_individual

    def get_elapsed_time(self) -> float:
        """The elapsed time since the start in seconds."""
        return (time_ns() - self.start_time) / 1_000_000  # seconds

    def get_number_evaluations(self) -> int:
        """The cumulative number of evaluations performed."""
        return self.evaluator.number_of_evaluations()

    def get_problem(self) -> Problem:
        return self.problem
