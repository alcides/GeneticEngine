from __future__ import annotations


import numpy as np

from sklearn.metrics import f1_score

from examples.benchmarks.benchmark import Benchmark, example_run
from examples.benchmarks.datasets import get_banknote
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.grammar import Grammar
from geneticengine.problems import Problem
from geneticengine.problems import SingleObjectiveProblem

from geml.common import forward_dataset
from geml.grammars.ruleset_classification import make_grammar


class ClassificationBenchmark(Benchmark):

    def __init__(self, X, y, feature_names):
        self.setup_problem(X, y)
        self.setup_grammar(y, feature_names)

    def setup_problem(self, data, target):

        # Problem
        def fitness_function(ruleset):
            try:
                y_pred = forward_dataset(ruleset.to_numpy(), data)
                with np.errstate(all="ignore"):
                    return f1_score(target, y_pred)
            except ValueError:
                return -1

        self.problem = SingleObjectiveProblem(minimize=False, fitness_function=fitness_function, target=1)

    def setup_grammar(self, y, feature_names):
        # Grammar
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
    example_run(ClassificationBenchmark(data, target, features))
