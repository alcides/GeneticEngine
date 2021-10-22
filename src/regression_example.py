from typing import Annotated

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error


from geneticengine.algorithms.gp import GP, create_tournament
from geneticengine.grammars.sgp import Plus, Literal, Number, Mul, SafeDiv, Var
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.treebased import treebased_representation
from geneticengine.metahandlers.vars import VarRange

# Load Dataset
bunch = load_diabetes()

# Prepare Grammar
Var.__annotations__["name"] = Annotated[str, VarRange(bunch.feature_names)]
g = extract_grammar([Plus, Mul, SafeDiv, Literal, Var], Number)
print("Grammar: {}.".format(repr(g)))


def safediv(x, y):
    if y == 0:
        return 0.00001
    else:
        return x / y


feature_indices = {}
for i, n in enumerate(bunch.feature_names):
    feature_indices[n] = i


def evaluate(n: Number):
    if isinstance(n, Plus):
        return lambda line: evaluate(n.left)(line) + evaluate(n.right)(line)
    elif isinstance(n, Mul):
        return lambda line: evaluate(n.left)(line) * evaluate(n.right)(line)
    elif isinstance(n, SafeDiv):
        return lambda line: safediv(evaluate(n.left)(line), evaluate(n.right)(line))
    elif isinstance(n, Literal):
        return lambda _: n.val
    elif isinstance(n, Var):
        return lambda line: line[feature_indices[n.name]]


def fitness_function(n):
    X = bunch.data
    y = bunch.target

    f = evaluate(n)
    y_pred = np.apply_along_axis(f, 1, X)
    return mean_squared_error(y, y_pred)


alg = GP(
    g,
    treebased_representation,
    fitness_function,
    minimize=True,
)
(b, bf) = alg.evolve()
print(bf, b)
