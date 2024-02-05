from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import numpy as np
import pandas as pd

from geneticengine.algorithms.gp.simplegp import SimpleGP
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.grammar import Grammar
from geneticengine.problems import MultiObjectiveProblem
from geneticengine.problems import Problem
from geml.grammars.basic_math import Exp
from geml.grammars.basic_math import SafeDiv
from geml.grammars.basic_math import SafeLog
from geml.grammars.basic_math import SafeSqrt
from geml.grammars.basic_math import Sin
from geml.grammars.basic_math import Tanh
from geml.grammars.sgp import Minus
from geml.grammars.sgp import Mul
from geml.grammars.sgp import Number
from geml.grammars.sgp import Plus
from geml.grammars.sgp import Var
from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.grammar.metahandlers.vars import VarRange

# ===================================
# This is a simple example of normal regression using normal GP,
# with a lexicase selection algorithm as the parent selection and mse metric for measuring the fitness
# We used the Vladislavleva4 data stored in examples/data folder
# ===================================

DATASET_NAME = "Vladislavleva4"
DATA_FILE_TRAIN = f"examples/data/{DATASET_NAME}/Train.txt"
DATA_FILE_TEST = f"examples/data/{DATASET_NAME}/Test.txt"

bunch = pd.read_csv(DATA_FILE_TRAIN, delimiter="\t")
target = bunch.response
data = bunch.drop(["response"], axis=1)

feature_names = list(data.columns.values)
feature_indices = {}
for i, n in enumerate(feature_names):
    feature_indices[n] = i

# Prepare Grammar
Var.__init__.__annotations__["name"] = Annotated[str, VarRange(feature_names)]
Var.feature_indices = feature_indices  # type: ignore


@dataclass
class Literal(Number):
    val: Annotated[int, IntRange(0, 9)]

    def evaluate(self, **kwargs):
        return self.val

    def __str__(self) -> str:
        return str(self.val)


def lexicase_parameters():
    X = data.values
    n_cases = 50
    case_size = int(len(X) / n_cases)
    if len(X) % case_size == 0:
        minimize_list = [True for _ in range(n_cases)]
    else:
        minimize_list = [True for _ in range(n_cases + 1)]

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

    def lexicase_fitness_function(n: Number):
        X = data.values
        y = target.values
        variables = {x: X[:, feature_indices[x]] for x in feature_names}

        try:
            y_pred = n.evaluate(**variables)
            pred_error = np.power(y_pred - y, 2)
            grouped_errors = calculate_grouped_errors(pred_error, n_cases, case_size)

            if len(X) % case_size != 0:
                last_case_fitness = calculate_case_fitness(pred_error, n_cases, case_size)
                grouped_errors.append(last_case_fitness)

        except (OverflowError, ValueError):
            return np.full(len(y), 99999999999)

        return grouped_errors

    return lexicase_fitness_function, minimize_list


class LexicaseRegressionBenchmark:
    def get_problem(self) -> Problem:
        fitness_function_lexicase, minimizelist = lexicase_parameters()
        return MultiObjectiveProblem(
            minimize=minimizelist,
            fitness_function=fitness_function_lexicase,
        )

    def get_grammar(self) -> Grammar:
        """<e>  ::=  <e>+<e>|

        <e>-<e>|       <e>*<e>|       pdiv(<e>,<e>)| psqrt(<e>)|
        np.sin(<e>)|       np.tanh(<e>)| np.exp(<e>)|       plog(<e>)|
        x[:, 0]|x[:, 1]|x[:, 2]|x[:, 3]|x[:, 4]|       <c> <c>  ::= 0 |
        1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
        """
        return extract_grammar(
            [
                Plus,
                Minus,
                Mul,
                SafeDiv,
                Literal,
                Var,
                SafeSqrt,
                Exp,
                Sin,
                Tanh,
                SafeLog,
            ],
            Number,
        )

    def main(self, **args):
        g = self.get_grammar()
        prob = self.get_problem()
        alg = SimpleGP(
            g,
            problem=prob,
            number_of_generations=10,
            selection_method=("lexicase", 0.01),
            **args,
        )
        best = alg.evolve()
        print(
            f"Fitness of {best.get_fitness(prob)} by genotype: {best.genotype} with phenotype: {best.get_phenotype()}",
        )


if __name__ == "__main__":
    LexicaseRegressionBenchmark().main(seed=0)
