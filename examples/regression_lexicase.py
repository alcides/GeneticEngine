from __future__ import annotations

from dataclasses import dataclass
from math import isinf
from typing import Annotated

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from geneticengine.algorithms.gp.gp_friendly import GPFriendly
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.problems import MultiObjectiveProblem
from geneticengine.core.representations.grammatical_evolution.ge import (
    GrammaticalEvolutionRepresentation,
)
from geneticengine.core.representations.grammatical_evolution.structured_ge import (
    StructuredGrammaticalEvolutionRepresentation,
)
from geneticengine.core.representations.tree.treebased import TreeBasedRepresentation
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

# ===================================
# This is a simple example of normal regression using normal GP,
# with a lexicase selection algorithm as the parent selection and mse metric for measuring the fitness
# We used the Vladislavleva4 data stored in examples/data folder
# ===================================

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


def fitness_function_lexicase(n: Number):
    assert isinstance(n, Number)
    cases = data.values.tolist()
    y = target.values

    fitnesses = list()

    for c in cases:
        variables = {}
        for x in feature_names:
            i = feature_indices[x]
            variables[x] = c[i]

        y_pred = n.evaluate(**variables)

        # mse is used in PonyGE, as the error metric is not None!
        fitness = mse(y_pred, y[cases.index(c)])
        if isinf(fitness) or np.isnan(fitness):
            fitness = 100000000

        fitnesses.append(fitness)

    return fitnesses


def evolve(
    g,
    seed,
    mode,
    representation="TreeBasedRepresentation",
):
    if representation == "ge":
        representation = GrammaticalEvolutionRepresentation
    elif representation == "sge":
        representation = GrammaticalEvolutionRepresentation
    else:
        representation = TreeBasedRepresentation

    minimizelist = [True for _ in data.values.tolist()]

    def single_criteria_test(n: Number) -> float:
        assert isinstance(n, Number)
        fitnesses = fitness_function_lexicase(n)
        return sum((m and -f or f) for (f, m) in zip(fitnesses, minimizelist))

    alg = GPFriendly(
        g,
        representation=representation,
        problem=MultiObjectiveProblem(
            minimize=minimizelist,
            fitness_function=fitness_function_lexicase,
            best_individual_criteria_function=single_criteria_test,
        ),
        probability_crossover=0.75,
        probability_mutation=0.01,
        number_of_generations=10,
        max_depth=8,
        population_size=50,
        selection_method=("lexicase",),
        n_elites=0,
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
