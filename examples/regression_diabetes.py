from __future__ import annotations

from typing import Annotated
from typing import Any
from typing import Callable

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error

from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.grammars.basic_math import Exp
from geneticengine.grammars.basic_math import SafeLog
from geneticengine.grammars.basic_math import SafeSqrt
from geneticengine.grammars.basic_math import Sin
from geneticengine.grammars.basic_math import Tanh
from geneticengine.grammars.sgp import Literal
from geneticengine.grammars.sgp import Mul
from geneticengine.grammars.sgp import Number
from geneticengine.grammars.sgp import Plus
from geneticengine.grammars.sgp import SafeDiv
from geneticengine.grammars.sgp import Var
from geneticengine.metahandlers.vars import VarRange

# Load Dataset
bunch: Any = load_diabetes()

feature_indices = {}
for i, n in enumerate(bunch.feature_names):
    feature_indices[n] = i

# Prepare Grammar
Var.__init__.__annotations__["name"] = Annotated[str, VarRange(bunch.feature_names)]
Var.feature_indices = feature_indices  # type: ignore


def preprocess():
    return extract_grammar(
        [Plus, Mul, SafeDiv, Literal, Var, SafeSqrt, Sin, Tanh, Exp, SafeLog],
        Number,
    )


def evolve(g, seed):
    alg = GP(
        g,
        fitness_function,
        treebased_representation,
        minimize=True,
        number_of_generations=10,
        seed=seed,
    )
    (b, bf, bp) = alg.evolve(verbose=1)
    return b, bf


def fitness_function(n: Number):
    X = bunch.data
    y = bunch.target

    variables = {}
    for x in bunch.feature_names:
        i = feature_indices[x]
        variables[x] = X[:, i]

    y_pred = n.evaluate(**variables)
    return mean_squared_error(y, y_pred, squared=False)


if __name__ == "__main__":
    g = preprocess()
    print(f"Grammar: {repr(g)}.")
    b, bf = evolve(g, 0)
    print(b, bf)
