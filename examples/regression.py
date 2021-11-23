from typing import Annotated, Any, Callable

import numpy as np
import pandas as pd
# from sklearn.metrics import mean_squared_error

from geneticengine.algorithms.gp.gp import GP
from geneticengine.grammars.sgp import Plus, Literal, Number, Mul, SafeDiv, Var
from geneticengine.grammars.math import SafeLog, SafeSqrt, Sin, Tanh, Exp
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.treebased import treebased_representation
from geneticengine.metahandlers.vars import VarRange
from geneticengine.metrics.metrics import rmse


FILE_NAME = "Vladislavleva4"
DATA_FILE_TRAIN = "./examples/data/{}/Train.txt".format(FILE_NAME)
DATA_FILE_TEST = "./examples/data/{}/Test.txt".format(FILE_NAME)

bunch = pd.read_csv(DATA_FILE_TRAIN,delimiter='\t')
target = bunch.response
data = bunch.drop(["response"],axis=1)

feature_names = list(data.columns.values)
feature_indices = {}
for i, n in enumerate(feature_names):
    feature_indices[n] = i


# Prepare Grammar
Var.__annotations__["name"] = Annotated[str, VarRange(feature_names)]
Var.feature_indices = feature_indices

def preprocess():
    return extract_grammar(
        [Plus, Mul, SafeDiv, Literal, Var, SafeSqrt, Exp, Sin, Tanh, SafeLog], Number)


def fitness_function(n: Number):
    X = data.values
    y = target.values

    f = n.evaluate_lines()
    y_pred = np.apply_along_axis(f, 1, X)
    return rmse(y_pred, y)

def evolve(g, seed):
    alg = GP(
        g,
        treebased_representation,
        fitness_function,
        minimize=True,
        number_of_generations=10,
        seed=seed,
    )
    (b, bf, bp) = alg.evolve(verbose=0)
    return b, bf

if __name__ == '__main__':
    g = preprocess()
    print("Grammar: {}.".format(repr(g)))
    b, bf = evolve(g, 0)
    print(b, bf)
