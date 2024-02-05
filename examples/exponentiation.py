from __future__ import annotations

from dataclasses import dataclass
from math import isinf
from typing import Annotated

import numpy as np
import pandas as pd

from geneticengine.algorithms.gp.simplegp import SimpleGP
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.grammar import Grammar
from geneticengine.problems import Problem
from geneticengine.problems import SingleObjectiveProblem
from geml.grammars.basic_math import SafeDiv
from geml.grammars.basic_math import SafeLog
from geml.grammars.basic_math import SafeSqrt
from geml.grammars.sgp import Literal
from geml.grammars.sgp import Minus
from geml.grammars.sgp import Mul
from geml.grammars.sgp import Number
from geml.grammars.sgp import Plus
from geml.grammars.sgp import Var
from geneticengine.grammar.metahandlers.vars import VarRange
from geml.metrics import mse

DATASET_NAME = "Vladislavleva4"
DATA_FILE_TRAIN = f"examples/data/{DATASET_NAME}/Train.txt"
DATA_FILE_TEST = f"examples/data/{DATASET_NAME}/Test.txt"

# ===================================
# This is a simple example on how to use GeneticEngine to solve a GP problem.
# We define the tree structure of the representation and then we define the fitness function for our problem
# We used the Vladislavleva4 dataset stored in examples/data folder
# ===================================

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
class Exponentiation(Number):
    baseNumber: Number
    powerNumber: Number

    def evaluate(self, **kwargs):
        d1 = self.baseNumber.evaluate(**kwargs)
        d2 = self.powerNumber.evaluate(**kwargs)
        try:
            return d1**d2
        except Exception:
            return 1.0


def fitness_function(n: Number):
    X = data.values
    y = target.values

    variables = {}
    for x in feature_names:
        i = feature_indices[x]
        variables[x] = X[:, i]
    try:
        y_pred = n.evaluate(**variables)
    except OverflowError:
        return 100000000
    fitness = mse(y_pred, y)
    if isinf(fitness) or np.isnan(fitness):
        fitness = 100000000
    return fitness


class ExponentiationBenchmark:
    def get_problem(self) -> Problem:
        return SingleObjectiveProblem(
            minimize=False,
            fitness_function=fitness_function,
        )

    def get_grammar(self) -> Grammar:
        return extract_grammar(
            [
                Plus,
                Minus,
                Mul,
                SafeDiv,
                Literal,
                Var,
                SafeSqrt,
                Exponentiation,
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
            max_depth=8,
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
    ExponentiationBenchmark().main(seed=0)
