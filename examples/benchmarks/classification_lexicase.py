from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from examples.benchmarks.benchmark import Benchmark, example_run
from geneticengine.grammar.grammar import extract_grammar, Grammar
from geneticengine.problems import Problem, MultiObjectiveProblem
from geml.common import forward_dataset
from geml.grammars.ruleset_classification import make_grammar


class ClassificationLexicaseBenchmark(Benchmark):
    def __init__(self, dataset_name="Banknote"):
        DATA_FILE_TRAIN = f"examples/data/{dataset_name}/Train.csv"

        bunch = pd.read_csv(DATA_FILE_TRAIN, delimiter=" ")
        target = bunch.y
        data = bunch.drop(["y"], axis=1)

        feature_names = list(data.columns.values)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data.values,
            target.values,
            test_size=0.75,
        )

        self.setup_problem()
        self.setup_grammar(feature_names)

    def setup_problem(self):
        def fitness_function_lexicase(ruleset):
            try:
                y_pred = forward_dataset(ruleset.to_numpy(), self.X_test)
                return [int(p == r) for (p, r) in zip(self.y_test, y_pred)]
            except ValueError:
                return [0 for _ in self.y_test]

        self.problem = MultiObjectiveProblem(minimize=False, fitness_function=fitness_function_lexicase)

    def setup_grammar(self, feature_names):
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
    example_run(ClassificationLexicaseBenchmark("Banknote"))
