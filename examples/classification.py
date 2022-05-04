from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from math import isinf
from typing import Annotated
from typing import Any
from typing import Callable

import numpy as np
import pandas as pd

from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.decorators import abstract
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.grammatical_evolution import ge_representation
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.grammars.basic_math import SafeDiv
from geneticengine.grammars.basic_math import SafeLog
from geneticengine.grammars.basic_math import SafeSqrt
from geneticengine.grammars.sgp import Mul
from geneticengine.grammars.sgp import Number
from geneticengine.grammars.sgp import Plus
from geneticengine.grammars.sgp import Var
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.vars import VarRange
from geneticengine.metrics import f1_score

DATASET_NAME = "Banknote"
DATA_FILE_TRAIN = f"examples/data/{DATASET_NAME}/Train.csv"
DATA_FILE_TEST = f"examples/data/{DATASET_NAME}/Test.csv"

bunch = pd.read_csv(DATA_FILE_TRAIN, delimiter=" ")
# import IPython as ip
# ip.embed()
target = bunch.y
data = bunch.drop(["y"], axis=1)

feature_names = list(data.columns.values)
feature_indices = {}
for i, n in enumerate(feature_names):
    feature_indices[n] = i

# Prepare Grammar
Var.__init__.__annotations__["name"] = Annotated[str, VarRange(feature_names)]
Var.feature_indices = feature_indices  # type: ignore


@abstract
class Literal(Number):
    pass


@dataclass
class One(Literal):
    def evaluate(self, **kwargs):
        return 1

    def __str__(self) -> str:
        return str(1)


@dataclass
class PointOne(Literal):
    def evaluate(self, **kwargs):
        return 0.1

    def __str__(self) -> str:
        return str(0.1)


@dataclass
class PointtOne(Literal):
    def evaluate(self, **kwargs):
        return 0.01

    def __str__(self) -> str:
        return str(0.01)


@dataclass
class PointttOne(Literal):
    def evaluate(self, **kwargs):
        return 0.001

    def __str__(self) -> str:
        return str(0.001)


@dataclass
class MinusPointttOne(Literal):
    def evaluate(self, **kwargs):
        return -0.001

    def __str__(self) -> str:
        return str(-0.001)


@dataclass
class MinusPointtOne(Literal):
    def evaluate(self, **kwargs):
        return -0.01

    def __str__(self) -> str:
        return str(-0.01)


@dataclass
class MinusPointOne(Literal):
    def evaluate(self, **kwargs):
        return -0.1

    def __str__(self) -> str:
        return str(-0.1)


@dataclass
class MinusOne(Literal):
    def evaluate(self, **kwargs):
        return -1

    def __str__(self) -> str:
        return str(-1)


literals = [
    MinusOne,
    MinusPointOne,
    MinusPointtOne,
    MinusPointttOne,
    One,
    PointOne,
    PointtOne,
    PointttOne,
]


def preprocess():
    return extract_grammar(
        [Plus, Mul, SafeDiv, Literal, Var, SafeSqrt, SafeLog] + literals,
        Number,
    )


# <e> ::= (<e> <op> <e>) | <f1>(<e>) | <f2>(<e>, <e>) | <v> | <c>
# <op> ::= + | * | -
# <f1> ::= psqrt | plog
# <f2> ::= pdiv
# <v> ::= x[:, <idx>]
# <idx> ::= 0 | 1 | 2 | 3
# <c> ::= -1.0 | -0.1 | -0.01 | -0.001 | 0.001 | 0.01 | 0.1 | 1.0


def fitness_function(n: Number):
    X = data.values
    y = target.values

    variables = {}
    for x in feature_names:
        i = feature_indices[x]
        variables[x] = X[:, i]
    y_pred = n.evaluate(**variables)

    if type(y_pred) in [np.float64, int, float]:
        """If n does not use variables, the output will be scalar."""
        y_pred = np.full(len(y), y_pred)
    if y_pred.shape != (len(y),):
        return -100000000
    fitness = f1_score(y_pred, y)
    if isinf(fitness):
        fitness = -100000000
    return fitness


def evolve(
    g,
    seed,
    mode,
    representation="treebased_representation",
    save_file="classification_evolution.csv",
):
    if representation == "grammatical_evolution":
        representation = ge_representation
    else:
        representation = treebased_representation

    alg = GP(
        g,
        fitness_function,
        representation=representation,
        selection_method=("tournament", 2),
        minimize=False,
        timer_stop_criteria=mode,
        args=sys.argv,
    )
    (b, bf, bp) = alg.evolve(verbose=1)
    return b, bf


if __name__ == "__main__":
    g = preprocess()
    print(g)
    b, bf = evolve(g, 123, False)
    print(bf)
    print(f"With fitness: {b}")
