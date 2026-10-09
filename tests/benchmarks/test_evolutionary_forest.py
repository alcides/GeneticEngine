from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from examples.benchmarks.evolutionary_forest import EvolutionaryForest, EvolutionaryForestBenchmark
from geml.grammars.symbolic_regression import Expression, Plus


@dataclass
class FixedVariable(Expression):
    name: str

    def to_numpy(self):
        return "dataset[:, 0]"

    def to_sympy(self):
        return self.name


def test_forest_averages_member_predictions():
    forest = EvolutionaryForest([FixedVariable("x"), Plus(FixedVariable("x"), FixedVariable("x"))])
    prediction = eval(forest.to_numpy(), {"np": np, "dataset": np.array([[2.0], [4.0]])})
    assert np.allclose(prediction, [3.0, 6.0])


def test_evolutionary_forest_benchmark_builds_and_scores():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 2.0])
    benchmark = EvolutionaryForestBenchmark(X, y, ["x"])
    assert benchmark.get_problem()
    assert benchmark.get_grammar()
