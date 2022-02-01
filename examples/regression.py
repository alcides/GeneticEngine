from typing import Annotated, Any, Callable

import os
import numpy as np
import pandas as pd
from math import isinf

from geneticengine.algorithms.gp.gp import GP
from geneticengine.grammars.sgp import Plus, Literal, Number, Mul, Var
from geneticengine.grammars.basic_math import SafeLog, SafeSqrt, Sin, Tanh, Exp, SafeDiv
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.core.representations.grammatical_evolution import ge_representation
from geneticengine.metahandlers.vars import VarRange
from geneticengine.metrics import rmse

DATASET_NAME = "Vladislavleva4"
DATA_FILE_TRAIN = "examples/data/{}/Train.txt".format(DATASET_NAME)
DATA_FILE_TEST = "examples/data/{}/Test.txt".format(DATASET_NAME)

bunch = pd.read_csv(DATA_FILE_TRAIN, delimiter="\t")
target = bunch.response
data = bunch.drop(["response"], axis=1)

feature_names = list(data.columns.values)
feature_indices = {}
for i, n in enumerate(feature_names):
    feature_indices[n] = i

# Prepare Grammar
Var.__annotations__["name"] = Annotated[str, VarRange(feature_names)]
Var.feature_indices = feature_indices  # type: ignore


def preprocess():
    return extract_grammar(
        [Plus, Mul, SafeDiv, Literal, Var, SafeSqrt, Exp, Sin, Tanh, SafeLog], Number
    )


def fitness_function(n: Number):
    X = data.values
    y = target.values

    variables = {}
    for x in feature_names:
        i = feature_indices[x]
        variables[x] = X[:, i]

    y_pred = n.evaluate(**variables)
    fitness = rmse(y_pred, y)
    if isinf(fitness) or np.isnan(fitness):
        fitness = 100000000
    return fitness


def evolve(g, seed, mode, representation='treebased_representation', output_folder=('','all')):
    if representation == 'grammatical_evolution':
        representation = ge_representation
    else:
        representation = treebased_representation
    
    alg = GP(
        g,
        fitness_function,
        representation=representation,
        minimize=True,
        selection_method=("tournament", 2),
        number_of_generations=150,
        seed=seed,
        timer_stop_criteria=mode,
        safe_gen_to_csv=output_folder
    )
    (b, bf, bp) = alg.evolve(verbose=0)
    return b, bf


if __name__ == "__main__":
    g = preprocess()
    bf, b = evolve(g, 0, False)
    print(b)
    print(f"With fitness: {bf}")
