from __future__ import annotations

import numpy as np

from examples.benchmarks.benchmark import Benchmark, example_run
from examples.benchmarks.datasets import get_banknote
from geneticengine.grammar.grammar import extract_grammar, Grammar
from geneticengine.problems import MultiObjectiveProblem, Problem
from geml.common import forward_dataset
from geml.grammars.ruleset_classification import make_grammar


class ClassificationLexicaseBenchmark(Benchmark):

    def __init__(self, X, y, feature_names):
        self.setup_problem(X, y)
        self.setup_grammar(y, feature_names)

    def setup_problem(self, X, y):
        def fitness_function_lexicase(ruleset):
            try:
                y_pred = forward_dataset(ruleset.to_numpy(), X)
                return np.equal(y, y_pred).reshape(len(y)).astype(int)
            except ValueError:
                return [0 for _ in y]

        n_objectives = len(y)
        self.problem = MultiObjectiveProblem(
            fitness_function=fitness_function_lexicase,
            minimize=[False for _ in range(n_objectives)],
            target=[1 for _ in range(n_objectives)],
        )

    def setup_grammar(self, y, feature_names):
        options = [int(v) for v in np.unique(y)]
        components, RuleSet = make_grammar(feature_names, options)
        Var = components[-1]
        Var.feature_names = feature_names
        index_of = {n: i for i, n in enumerate(feature_names)}
        Var.to_numpy = lambda s: f"dataset[:,{index_of[s.name]}]"
        self.grammar = extract_grammar(components, RuleSet)

    def get_problem(self) -> Problem:
        return self.problem

    def get_grammar(self) -> Grammar:
        return self.grammar


if __name__ == "__main__":
    data, target, features = get_banknote()
    example_run(ClassificationLexicaseBenchmark(data, target, features))
