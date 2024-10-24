from __future__ import annotations


import numpy as np

from sklearn.metrics import r2_score

from examples.benchmarks.benchmark import Benchmark, example_run
from examples.benchmarks.datasets import get_vladislavleva
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.grammar import Grammar
from geneticengine.problems import Problem
from geneticengine.problems import SingleObjectiveProblem

from geml.common import forward_dataset
from geml.grammars.symbolic_regression import Expression, components, make_var


class RegressionBenchmark(Benchmark):

    def __init__(self, X, y, feature_names):
        self.setup_problem(X, y)
        self.setup_grammar(feature_names)

    def setup_problem(self, data, target):

        # Problem
        def fitness_function(ruleset):
            try:
                y_pred = forward_dataset(ruleset.to_numpy(), data)
                with np.errstate(all="ignore"):
                    return r2_score(target, y_pred)
            except ValueError:
                return -1

        self.problem = SingleObjectiveProblem(minimize=False, fitness_function=fitness_function, target=0)

    def setup_grammar(self, feature_names):
        # Grammar
        Var = make_var(feature_names)
        grammar_components = components + [Var]
        index_of = {n: i for i, n in enumerate(feature_names)}
        Var.to_numpy = lambda s: f"dataset[:,{index_of[s.name]}]"
        self.grammar = extract_grammar(grammar_components, Expression)

    def get_problem(self) -> Problem:
        return self.problem

    def get_grammar(self) -> Grammar:
        return self.grammar


if __name__ == "__main__":
    X, y, feature_names = get_vladislavleva()
    example_run(RegressionBenchmark(X, y, feature_names))
