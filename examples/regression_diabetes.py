from __future__ import annotations

from math import isinf
from typing import Annotated
from typing import Any

import numpy as np
from sklearn.datasets import load_diabetes

from geneticengine.algorithms.gp.simplegp import SimpleGP
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.grammar import Grammar
from geneticengine.core.problems import Problem
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.grammars.basic_math import Exp
from geneticengine.grammars.basic_math import SafeDiv
from geneticengine.grammars.basic_math import SafeLog
from geneticengine.grammars.basic_math import SafeSqrt
from geneticengine.grammars.basic_math import Sin
from geneticengine.grammars.basic_math import Tanh
from geneticengine.grammars.sgp import Literal
from geneticengine.grammars.sgp import Mul
from geneticengine.grammars.sgp import Number
from geneticengine.grammars.sgp import Plus
from geneticengine.grammars.sgp import Var
from geneticengine.metahandlers.vars import VarRange
from geneticengine.metrics import mse

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
            target_fitness=None,
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
        best = alg.evolve()
        print(
            f"Fitness of {prob.overall_fitness(best.get_phenotype())} by genotype: {best.genotype} with phenotype: {best.get_phenotype()}",
        )


if __name__ == "__main__":
    DiabetesBenchmark().main(seed=0)
