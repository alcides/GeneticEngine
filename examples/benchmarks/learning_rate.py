"""Dependency-free adaptive learning-rate benchmark.

This is a small GeneticEngine analogue of the AutoLR task: evolve a schedule
of learning rates that follows the rates required by a synthetic optimization
problem. It is intentionally cheap enough to run as an example and test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import numpy as np

from examples.benchmarks.benchmark import Benchmark, example_run
from geneticengine.grammar.grammar import Grammar, extract_grammar
from geneticengine.grammar.metahandlers.floats import FloatRange
from geneticengine.grammar.metahandlers.lists import ListSizeBetween
from geneticengine.problems import Problem, SingleObjectiveProblem

EPOCHS = 10
TARGET_SCHEDULE = np.geomspace(0.1, 0.001, EPOCHS)


@dataclass
class LearningRateSchedule:
    rates: Annotated[list[Annotated[float, FloatRange(0.0001, 0.2)]], ListSizeBetween(EPOCHS, EPOCHS)]

    def values(self) -> np.ndarray:
        return np.asarray(self.rates, dtype=float)


class LearningRateBenchmark(Benchmark):
    def __init__(self):
        self.grammar = extract_grammar([LearningRateSchedule], LearningRateSchedule)
        self.problem = SingleObjectiveProblem(minimize=True, target=0.0, fitness_function=self.fitness)

    def fitness(self, schedule: LearningRateSchedule) -> float:
        return float(np.mean(np.square(schedule.values() - TARGET_SCHEDULE)))

    def get_problem(self) -> Problem:
        return self.problem

    def get_grammar(self) -> Grammar:
        return self.grammar


if __name__ == "__main__":
    example_run(LearningRateBenchmark())
