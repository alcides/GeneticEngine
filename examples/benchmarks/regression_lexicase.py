from __future__ import annotations


import numpy as np


from examples.benchmarks.benchmark import Benchmark, example_run
from examples.benchmarks.datasets import get_vladislavleva
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.grammar import Grammar
from geneticengine.problems import MultiObjectiveProblem, Problem

from geml.common import forward_dataset
from geml.grammars.symbolic_regression import Expression, components, make_var


class RegressionLexicaseBenchmark(Benchmark):

    def __init__(self, X, y, feature_names):
        self.setup_problem(X, y)
        self.setup_grammar(feature_names)

    def setup_problem(self, data, target):

        # Problem
        X = data
        y = target
        n_cases = 50
        case_size = int(len(data) / n_cases)

        def calculate_case_fitness(pred_error, i, case_size):
            start_index = case_size * i
            end_index = case_size * (i + 1)
            case_error = pred_error[start_index:end_index]
            case_fitness = sum(case_error) / len(case_error)

            if np.isinf(case_fitness) or np.isnan(case_fitness):
                case_fitness = 100000000
            return case_fitness

        def calculate_grouped_errors(pred_error, n_cases, case_size):
            grouped_errors = []
            for i in range(n_cases):
                case_fitness = calculate_case_fitness(pred_error, i, case_size)
                grouped_errors.append(case_fitness)
            return grouped_errors

        def lexicase_fitness_function(n: Expression):

            try:
                y_pred = forward_dataset(n.to_numpy(), X)
                pred_error = np.power(y_pred - y, 2)
                grouped_errors = calculate_grouped_errors(pred_error, n_cases, case_size)

                if len(X) % case_size != 0:
                    last_case_fitness = calculate_case_fitness(pred_error, n_cases, case_size)
                    grouped_errors.append(last_case_fitness)

            except (OverflowError, ValueError):
                return np.full(len(y), 99999999999)

            return grouped_errors

        n_objectives = n_cases
        self.problem = MultiObjectiveProblem(
            fitness_function=lexicase_fitness_function,
            minimize=[True for _ in range(n_objectives)],
            target=[0 for _ in range(n_objectives)],
        )

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
    example_run(RegressionLexicaseBenchmark(X, y, feature_names))
