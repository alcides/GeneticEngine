from __future__ import annotations

import os
from dataclasses import dataclass
from math import isinf
from typing import Annotated

import numpy as np
import pandas as pd

from geneticengine.algorithms.gp.gp_friendly import GPFriendly
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.representations.grammatical_evolution.ge import (
    GrammaticalEvolutionRepresentation,
)
from geneticengine.core.representations.grammatical_evolution.structured_ge import (
    StructuredGrammaticalEvolutionRepresentation,
)
from geneticengine.core.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.grammars.basic_math import SafeDiv
from geneticengine.grammars.basic_math import SafeLog
from geneticengine.grammars.basic_math import SafeSqrt
from geneticengine.grammars.sgp import Literal
from geneticengine.grammars.sgp import Minus
from geneticengine.grammars.sgp import Mul
from geneticengine.grammars.sgp import Number
from geneticengine.grammars.sgp import Plus
from geneticengine.grammars.sgp import Var
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.vars import VarRange
from geneticengine.metrics import mse

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
        except:
            return 1.0


def preprocess():
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


def fitness_function(n: Number):
    X = data.values
    y = target.values

    variables = {}
    for x in feature_names:
        i = feature_indices[x]
        variables[x] = X[:, i]

    y_pred = n.evaluate(**variables)
    fitness = mse(y_pred, y)
    if isinf(fitness) or np.isnan(fitness):
        fitness = 100000000
    return fitness


def evolve(
    g,
    seed,
    mode,
):
    alg = GPFriendly(
        g,
        representation=TreeBasedRepresentation,
        problem=SingleObjectiveProblem(
            minimize=True,
            fitness_function=fitness_function,
            target_fitness=None,
        ),
        probability_crossover=0.75,
        probability_mutation=0.01,
        number_of_generations=50,
        max_depth=8,
        population_size=50,
        selection_method=("tournament", 2),
        n_elites=5,
        seed=seed,
        timer_stop_criteria=mode,
    )
    (b, bf, bp) = alg.evolve()
    return b, bf


if __name__ == "__main__":
    g = preprocess()
    bf, b = evolve(g, 0, False)
    print(b)
    print(f"With fitness: {bf}")
