from typing import Annotated, Any, Callable

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error

from geneticengine.algorithms.gp.gp import GP
from geneticengine.grammars.sgp import Plus, Literal, Number, Mul, SafeDiv, Var
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.treebased import treebased_representation
from geneticengine.metahandlers.vars import VarRange

# Load Dataset
bunch: Any = load_diabetes()

feature_indices = {}
for i, n in enumerate(bunch.feature_names):
    feature_indices[n] = i

# Prepare Grammar
Var.__annotations__["name"] = Annotated[str, VarRange(bunch.feature_names)]
Var.feature_indices = feature_indices
g = extract_grammar([Plus, Mul, Literal, Var], Number)
print("Grammar: {}.".format(repr(g)))


def fitness_function(n: Number):
    X = bunch.data
    y = bunch.target

    f = n.evaluate_lines()
    y_pred = np.apply_along_axis(f, 1, X)
    return mean_squared_error(y, y_pred)


alg = GP(
    g,
    treebased_representation,
    fitness_function,
    minimize=True,
    number_of_generations=10,
)
(b, bf, bp) = alg.evolve(verbose=0)
print(bf, bp, b)
