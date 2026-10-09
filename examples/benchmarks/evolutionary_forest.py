"""A compact Evolutionary Forest-style symbolic regression benchmark.

Each individual is an ensemble of independently evolved symbolic regression
trees. The ensemble prediction is the mean of its members, matching the core
idea of Evolutionary Forest while keeping the example self-contained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import numpy as np

from examples.benchmarks.benchmark import Benchmark, example_run
from examples.benchmarks.datasets import get_vladislavleva
from geml.common import forward_dataset
from geml.grammars.symbolic_regression import Expression, components, make_var
from geneticengine.grammar.grammar import Grammar, extract_grammar
from geneticengine.grammar.metahandlers.lists import ListSizeBetween
from geneticengine.problems import Problem, SingleObjectiveProblem


@dataclass
class EvolutionaryForest:
    # The reference EvolutionaryForest estimator defaults to ensemble_size=100.
    trees: Annotated[list[Expression], ListSizeBetween(100, 100)]

    def to_numpy(self) -> str:
        predictions = ", ".join(tree.to_numpy() for tree in self.trees)
        return f"np.mean(np.array([{predictions}]), axis=0)"


class EvolutionaryForestBenchmark(Benchmark):
    def __init__(self, X, y, feature_names: list[str]):
        self.X = X
        self.y = y
        variables = make_var(feature_names)
        index_of = {name: index for index, name in enumerate(feature_names)}
        variables.to_numpy = lambda variable: f"dataset[:, {index_of[variable.name]}]"
        self.grammar = extract_grammar([EvolutionaryForest, *components, variables], EvolutionaryForest)
        self.problem = SingleObjectiveProblem(
            minimize=True,
            target=0.0,
            fitness_function=self.fitness,
        )

    def fitness(self, forest: EvolutionaryForest) -> float:
        try:
            prediction = forward_dataset(forest.to_numpy(), self.X)
            with np.errstate(all="ignore"):
                error = np.mean(np.square(self.y - prediction))
            return float(error) if np.isfinite(error) else float("inf")
        except (ArithmeticError, TypeError, ValueError, ZeroDivisionError):
            return float("inf")

    def get_problem(self) -> Problem:
        return self.problem

    def get_grammar(self) -> Grammar:
        return self.grammar


if __name__ == "__main__":
    X, y, feature_names = get_vladislavleva()
    example_run(EvolutionaryForestBenchmark(X, y, feature_names))
