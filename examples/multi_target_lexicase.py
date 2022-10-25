from __future__ import annotations

from ctypes import sizeof
from dataclasses import dataclass
from math import isinf
from typing import Annotated

import numpy as np
import pandas as pd
from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split

from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.problems import MultiObjectiveProblem
from geneticengine.core.representations.grammatical_evolution.ge import (
    ge_representation,
)
from geneticengine.core.representations.grammatical_evolution.structured_ge import (
    sge_representation,
)
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.grammars.basic_math import Exp
from geneticengine.grammars.basic_math import SafeDiv
from geneticengine.grammars.basic_math import SafeLog
from geneticengine.grammars.basic_math import SafeSqrt
from geneticengine.grammars.basic_math import Sin
from geneticengine.grammars.basic_math import Tanh
from geneticengine.grammars.sgp import Minus
from geneticengine.grammars.sgp import Mul
from geneticengine.grammars.sgp import Number
from geneticengine.grammars.sgp import Plus
from geneticengine.grammars.sgp import Var
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.lists import ListSizeBetween
from geneticengine.metahandlers.vars import VarRange
from geneticengine.metrics import mse


# Load the data from Sklearn
bunch = load_linnerud(as_frame=True)  # returns a Bunch instance
X, y = bunch["data"], bunch["target"]


targets = y.values.tolist()

feature_names = bunch["feature_names"]
target_names = bunch["target_names"]

feature_indices = {}
for i, n in enumerate(feature_names):
    feature_indices[n] = i


Var.__init__.__annotations__["name"] = Annotated[str, VarRange(feature_names)]
Var.feature_indices = feature_indices  # type: ignore


@dataclass
class Literal(Number):
    val: Annotated[int, IntRange(0, 9)]

    def evaluate(self, **kwargs):
        return self.val

    def __str__(self) -> str:
        return str(self.val)


@dataclass
class NumberList:
    lst: Annotated[list[Number], ListSizeBetween(3, 3)]

    def evaluate(self, **kwargs):
        return [n.evaluate(**kwargs) for n in self.lst]

    def __str__(self) -> str:
        s = "["
        for i in range(len(self.lst)):
            s += str(self.lst[i])
            s += "," if i < len(self.lst) - 1 else "]"
        return s


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
            Exp,
            Sin,
            Tanh,
            SafeLog,
        ],
        NumberList,
    )


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

y_weight = y_train["Weight"].tolist()
y_waist = y_train["Waist"].tolist()
y_pulse = y_train["Pulse"].tolist()


def fitness_function_lexicase(n: Number):
    cases = []

    cases.append(X_train["Chins"].tolist())
    cases.append(X_train["Situps"].tolist())
    cases.append(X_train["Jumps"].tolist())

    fitnesses = list()

    def fitness_aux(index, y_target):
        y = y_target

        fit = list()

        for c in cases:
            variables = {}
            for x in feature_names:
                i = feature_indices[x]
                variables[x] = c[i]

            # y_pred this wil be a list of 3 functions
            y_pred = n.evaluate(**variables)

            # mse is used in PonyGE, as the error metric is not None!
            fitness = mse(y_pred[index], y[cases.index(c)])
            if isinf(fitness) or np.isnan(fitness):
                fitness = 100000000

            fit.append(fitness)

        return sum(fit) / len(fit)

    fitnesses.append(fitness_aux(0, y_weight))
    fitnesses.append(fitness_aux(1, y_waist))
    fitnesses.append(fitness_aux(2, y_pulse))
    return fitnesses


def evolve(
    g,
    seed,
    mode,
    representation="treebased_representation",
):
    if representation == "ge":
        representation = ge_representation
    elif representation == "sge":
        representation = sge_representation
    else:
        representation = treebased_representation

    minimizelist = [True for _ in X.values.tolist()]

    alg = GP(
        g,
        representation=representation,
        problem=MultiObjectiveProblem(
            minimize=minimizelist,
            fitness_function=fitness_function_lexicase,
        ),
        # As in PonyGE2:
        probability_crossover=0.75,
        probability_mutation=0.01,
        number_of_generations=50,
        max_depth=15,
        population_size=50,
        selection_method=("lexicase",),
        n_elites=0,
        # ----------------
        seed=seed,
        timer_stop_criteria=mode,
    )
    (b, bf, bp) = alg.evolve(verbose=1)
    return b, bf


if __name__ == "__main__":
    g = preprocess()
    bf, b = evolve(g, 0, False)
    print(b)
    print(f"With fitness: {bf}")
