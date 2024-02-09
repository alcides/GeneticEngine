from __future__ import annotations

from math import isinf
from typing import Annotated
from typing import Any

import numpy as np
from sklearn.datasets import load_diabetes

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
from geml.grammars.sgp import Literal
from geml.grammars.sgp import Mul
from geml.grammars.sgp import Number
from geml.grammars.sgp import Plus
from geml.grammars.sgp import Var
from geneticengine.grammar.metahandlers.vars import VarRange
from geml.metrics import mse

# ===================================
# This is a simple example of normal regression using normal GP,
# with a tournament selection algorithm as the parent selection and mse metric for measuring the fitness
# We used the diabetes dataset from sklearn library
# ===================================

# Load Dataset
bunch: Any = load_diabetes()

feature_indices = {}
for i, n in enumerate(bunch.feature_names):
    feature_indices[n] = i

# Prepare Grammar
Var.__init__.__annotations__["name"] = Annotated[str, VarRange(bunch.feature_names)]
Var.feature_indices = feature_indices  # type: ignore


def fitness_function(n: Number):
    X = bunch.data
    y = bunch.target

    variables = {}
    for x in bunch.feature_names:
        i = feature_indices[x]
        variables[x] = X[:, i]

    y_pred = n.evaluate(**variables)
    # mse is used in PonyGE, as the error metric is not None!
    fitness = mse(y_pred, y)
    if isinf(fitness) or np.isnan(fitness):
        fitness = 100000000
    return fitness


class DiabetesBenchmark:
    def get_problem(self) -> Problem:
        return SingleObjectiveProblem(
            minimize=True,
            fitness_function=fitness_function,
        )

    def get_grammar(self) -> Grammar:
        return extract_grammar(
            [Plus, Mul, SafeDiv, Literal, Var, SafeSqrt, Sin, Tanh, Exp, SafeLog],
            Number,
        )

    def main(self, **args):
        g = self.get_grammar()
        prob = self.get_problem()
        alg = SimpleGP(
            g,
            problem=prob,
            number_of_generations=10,
            **args,
        )
        best = alg.search()
        print(
            f"Fitness of {best.get_fitness(prob)} by genotype: {best.genotype} with phenotype: {best.get_phenotype()}",
        )


if __name__ == "__main__":
    DiabetesBenchmark().main(seed=0)
