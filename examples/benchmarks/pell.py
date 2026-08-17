"""A benchmark for the positive solutions of Pell's equation.

The default instance is ``x² - 2y² = 1``.  Pell solutions are conventionally
positive; allowing arbitrary signed integers would make minimizing ``x + y``
unbounded below because the negation of every solution is also a solution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from examples.benchmarks.benchmark import Benchmark, example_run
from geneticengine.grammar.grammar import Grammar, extract_grammar
from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.problems import Problem, SingleObjectiveProblem


@dataclass
class PellPair:
    x: Annotated[int, IntRange(1, 100)]
    y: Annotated[int, IntRange(1, 100)]

    def evaluate(self, d: int) -> tuple[int, int]:
        return self.x, self.y


def pell_fitness(candidate: PellPair, d: int = 2) -> float:
    """Penalize equation violations, then minimize the positive coordinate sum."""
    x, y = candidate.evaluate(d)
    residual = x * x - d * y * y - 1
    return abs(residual) * 1_000_000 + x + y


class PellsEquationBenchmark(Benchmark):
    """Find the smallest positive solution to ``x² - 2y² = 1``."""

    def __init__(self, d: int = 2) -> None:
        if d <= 0 or int(d**0.5) ** 2 == d:
            raise ValueError("d must be positive and nonsquare")
        self.d = d
        self.problem = SingleObjectiveProblem(
            minimize=True,
            target=5 if d == 2 else None,
            fitness_function=lambda candidate: pell_fitness(candidate, d),
        )
        self.grammar = extract_grammar([PellPair], PellPair)

    def get_problem(self) -> Problem:
        return self.problem

    def get_grammar(self) -> Grammar:
        return self.grammar


if __name__ == "__main__":
    example_run(PellsEquationBenchmark())
