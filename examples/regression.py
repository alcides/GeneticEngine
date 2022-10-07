from __future__ import annotations

import os
from dataclasses import dataclass
from math import isinf
from typing import Annotated
from typing import Any
from typing import Callable

import global_vars as gv
import numpy as np
import pandas as pd

from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.grammatical_evolution.dynamic_structured_ge import (
    dsge_representation,
)
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
from geneticengine.metahandlers.vars import VarRange
from geneticengine.metrics import mse

DATASET_NAME = "Vladislavleva4"
DATA_FILE_TRAIN = f"examples/data/{DATASET_NAME}/Train.txt"
DATA_FILE_TEST = f"examples/data/{DATASET_NAME}/Test.txt"

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
class Literal(Number):
    val: Annotated[int, IntRange(0, 9)]

    def evaluate(self, **kwargs):
        return self.val

    def __str__(self) -> str:
        return str(self.val)


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
        Number,
    )

    # <e>  ::=  <e>+<e>|
    #       <e>-<e>|
    #       <e>*<e>|
    #       pdiv(<e>,<e>)|
    #       psqrt(<e>)|
    #       np.sin(<e>)|
    #       np.tanh(<e>)|
    #       np.exp(<e>)|
    #       plog(<e>)|
    #       x[:, 0]|x[:, 1]|x[:, 2]|x[:, 3]|x[:, 4]|
    #       <c>
    # <c>  ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9


def fitness_function(n: Number):
    X = data.values
    y = target.values

    variables = {}
    for x in feature_names:
        i = feature_indices[x]
        variables[x] = X[:, i]

    y_pred = n.evaluate(**variables)
    # mse is used in PonyGE, as the error metric is not None!
    fitness = mse(y_pred, y)
    if isinf(fitness) or np.isnan(fitness):
        fitness = 100000000
    return fitness


def evolve(
    seed,
    mode,
    save_to_csv: str = None,
    representation="treebased_representation",
):
    if representation == "ge":
        representation = ge_representation
    elif representation == "sge":
        representation = sge_representation
    elif representation == "dsge":
        representation = dsge_representation
    else:
        representation = treebased_representation

    g = preprocess()
    alg = GP(
        g,
        fitness_function,
        representation=representation,
        probability_crossover=gv.PROBABILITY_CROSSOVER,
        probability_mutation=gv.PROBABILITY_MUTATION,
        number_of_generations=gv.NUMBER_OF_GENERATIONS,
        max_depth=gv.MAX_DEPTH,
        population_size=gv.POPULATION_SIZE,
        selection_method=gv.SELECTION_METHOD,
        n_elites=gv.N_ELITES,
        # ----------------
        minimize=False,
        seed=seed,
        timer_stop_criteria=mode,
        save_to_csv=save_to_csv,
        save_genotype_as_string=False,
    )
    (b, bf, bp) = alg.evolve(verbose=1)
    return b, bf
