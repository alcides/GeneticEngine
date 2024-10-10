from __future__ import annotations


import numpy as np
import pandas as pd

from sklearn.metrics import f1_score

from examples.benchmarks.benchmark import Benchmark, example_run
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.grammar import Grammar
from geneticengine.problems import Problem
from geneticengine.problems import SingleObjectiveProblem

from geml.common import forward_dataset
from geml.grammars.ruleset_classification import make_grammar


class ClassificationBenchmark(Benchmark):
    def __init__(self, dataset_name="Banknote"):
        DATA_FILE_TRAIN = f"examples/data/{dataset_name}/Train.csv"
        # DATA_FILE_TEST = f"examples/data/{dataset_name}/Test.csv"

        bunch = pd.read_csv(DATA_FILE_TRAIN, delimiter=" ")
        target = bunch.y
        data = bunch.drop(["y"], axis=1)

        feature_names = list(data.columns.values)

        self.setup_problem(data.values, target.values)
        self.setup_grammar(feature_names)

    def setup_problem(self, data, target):

        # Problem
        def fitness_function(ruleset):
            try:
                y_pred = forward_dataset(ruleset.to_numpy(), data)
                with np.errstate(all="ignore"):
                    return f1_score(target, y_pred)
            except ValueError:
                return -1

        self.problem = SingleObjectiveProblem(minimize=False, fitness_function=fitness_function)

    def setup_grammar(self, feature_names):
        # Grammar
        components, RuleSet = make_grammar(feature_names, [0, 1])
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
    example_run(ClassificationBenchmark("Banknote"))
