from __future__ import annotations

import os
import sys
from abc import ABC
from dataclasses import dataclass
from typing import Annotated
from typing import Any
from typing import Callable
from typing import Tuple

import numpy as np
from sklearn.metrics import f1_score

from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.grammatical_evolution import ge_representation
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.grammars.coding.classes import Condition
from geneticengine.grammars.coding.classes import Expr
from geneticengine.grammars.coding.classes import Number
from geneticengine.grammars.coding.logical_ops import And
from geneticengine.grammars.coding.logical_ops import Not
from geneticengine.grammars.coding.logical_ops import Or
from geneticengine.metahandlers.ints import IntRange

DATASET_NAME = "GameOfLife"
DATA_FILE_TRAIN = f"examples/data/{DATASET_NAME}/Train.csv"
DATA_FILE_TEST = f"examples/data/{DATASET_NAME}/Test.csv"
OUTPUT_FOLDER = "GoL/grammar_standard"


train = np.genfromtxt(DATA_FILE_TRAIN, skip_header=1, delimiter=",")
Xtrain = train[:, :-1]
Xtrain = Xtrain.reshape(train.shape[0], 3, 3)
ytrain = train[:, -1]

test = np.genfromtxt(DATA_FILE_TEST, skip_header=1, delimiter=",")
Xtest = test[:, :-1]
Xtest = Xtest.reshape(test.shape[0], 3, 3)
ytest = test[:, -1]


@dataclass
class MatrixElement(Condition):
    row: Annotated[int, IntRange(0, 2)]
    column: Annotated[int, IntRange(0, 2)]

    def __str__(self) -> str:
        return f"(X[{self.row}, {self.column}])"


def evaluate(e: Expr) -> Callable[[Any], float]:

    if isinstance(e, And):
        f1 = evaluate(e.left)
        f2 = evaluate(e.right)
        return lambda line: f1(line) and f2(line)
    elif isinstance(e, Or):
        f1 = evaluate(e.left)
        f2 = evaluate(e.right)
        return lambda line: f1(line) or f2(line)
    elif isinstance(e, Not):
        f1 = evaluate(e.cond)
        return lambda line: not f1(line)
    elif isinstance(e, MatrixElement):
        r = e.row
        c = e.column
        return lambda line: line[r, c]
    else:
        print(type(e), isinstance(e, MatrixElement))
        raise NotImplementedError(str(e), type(e))


def fitness_function(i: Expr):
    _clf = evaluate(i)
    ypred = [_clf(line) for line in np.rollaxis(Xtrain, 0)]
    return f1_score(ytrain, ypred)


def preprocess():
    grammar = extract_grammar([And, Or, Not, MatrixElement], Condition)
    print(grammar)
    return grammar


def evolve(
    g,
    mode,
    representation="treebased_representation",
):
    if representation == "grammatical_evolution":
        representation = ge_representation
    else:
        representation = treebased_representation

    alg = GP(
        g,
        fitness_function,
        representation=representation,
        # favor_less_deep_trees=True,
        # As in PonyGE2:
        selection_method=("tournament", 2),
        # ----------------
        minimize=False,
        timer_stop_criteria=mode,
        args=sys.argv,
    )
    (b, bf, bp) = alg.evolve(verbose=1)

    print("Best individual:", bp)
    print("Genetic Engine Train F1 score:", bf)

    _clf = evaluate(bp)
    ypred = [_clf(line) for line in np.rollaxis(Xtest, 0)]
    print("GeneticEngine Test F1 score:", f1_score(ytest, ypred))

    return b, bf


if __name__ == "__main__":
    g = preprocess()
    bf, b = evolve(g, 0, False)
    print(b)
    print(f"With fitness: {bf}")
