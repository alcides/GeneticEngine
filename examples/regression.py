from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import numpy as np
import pandas as pd

from geneticengine.algorithms.gp.simplegp import SimpleGP
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.grammar import Grammar
from geneticengine.problems import Problem
from geneticengine.problems import SingleObjectiveProblem
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
from geml.metrics import mse

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


def fitness_function(n: Number):
    X = data.values
    y = target.values

    variables = {}
    for x in feature_names:
        i = feature_indices[x]
        variables[x] = X[:, i]

    y_pred = n.evaluate(**variables)
    # mse is used in PonyGE, as the error metric is not None!
    fitness = mse(y_pred, y)
    if np.isinf(fitness) or np.isnan(fitness):
        fitness = 100000000
    return fitness


class RegressionBenchmark:
    def get_problem(self) -> Problem:
        return SingleObjectiveProblem(
            minimize=True,
            fitness_function=fitness_function,
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
            probability_crossover=0.75,
            probability_mutation=0.01,
            number_of_generations=50,
            max_depth=10,
            population_size=50,
            selection_method=("tournament", 2),
            n_elites=5,
            **args,
        )
        best = alg.evolve()
        print(
            f"Fitness of {best.get_fitness(prob)} by genotype: {best.genotype} with phenotype: {best.get_phenotype()}",
        )


if __name__ == "__main__":
    RegressionBenchmark().main(seed=0)
