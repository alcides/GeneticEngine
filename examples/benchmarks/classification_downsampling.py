from examples.benchmarks.datasets import get_banknote
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.problems import MultiObjectiveProblem
from geml.common import forward_dataset
from geml.grammars.ruleset_classification import make_grammar
from geneticengine.algorithms.gp.operators.selection import InformedDownsamplingSelection
from examples.benchmarks.benchmark import Benchmark

import numpy as np


class ClassificationDownsamplingBenchmark(Benchmark):
    def __init__(self, X, y, feature_names):
        self.setup_problem(X, y)
        self.setup_grammar(X, y, feature_names)

    def setup_problem(self, X, y):
        def fitness_function(ruleset):
            try:
                y_pred = forward_dataset(ruleset.to_numpy(), X)
                return np.equal(y, y_pred).reshape(len(y)).astype(int)
            except ValueError:
                return [0 for _ in y]

        n_objectives = len(y)
        self.problem = MultiObjectiveProblem(
            fitness_function=fitness_function,
            minimize=[False for _ in range(n_objectives)],
            target=[1 for _ in range(n_objectives)],
        )

    def setup_grammar(self, X, y, feature_names):
        options = [int(v) for v in np.unique(y)]
        components, RuleSet = make_grammar(feature_names, options)
        Var = components[-1]
        Var.feature_names = feature_names
        index_of = {f: i for i, f in enumerate(feature_names)}  # FIXED: use str not f.name
        Var.to_numpy = lambda s: f"dataset[:,{index_of[s.name]}]"
        self.grammar = extract_grammar(components, RuleSet)

    def get_problem(self):
        return self.problem

    def get_grammar(self):
        return self.grammar

    def get_selector(self):
        return InformedDownsamplingSelection(max_sample_size=10)

if __name__ == "__main__":
    data, target, features = get_banknote()
    print("=== Running InformedDownsampling Benchmark ===")
    from examples.benchmarks.benchmark import example_run
    example_run(ClassificationDownsamplingBenchmark(data, target, features))
