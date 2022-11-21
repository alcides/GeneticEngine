from __future__ import annotations

import os
from dataclasses import dataclass
from math import isinf
from typing import Annotated

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from geneticengine.algorithms.callbacks.pge import PGECallback
from geneticengine.algorithms.gp.simplegp import SimpleGP
from geneticengine.core.decorators import get_gengy
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.representations.grammatical_evolution.dynamic_structured_ge import (
    DynamicStructuredGrammaticalEvolutionRepresentation,
)
from geneticengine.core.representations.grammatical_evolution.ge import (
    GrammaticalEvolutionRepresentation,
)
from geneticengine.core.representations.grammatical_evolution.structured_ge import (
    StructuredGrammaticalEvolutionRepresentation,
)
from geneticengine.core.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.grammars.basic_math import SafeDiv
from geneticengine.grammars.basic_math import SafeLog
from geneticengine.grammars.basic_math import SafeSqrt
from geneticengine.grammars.sgp import Mul
from geneticengine.grammars.sgp import Number
from geneticengine.grammars.sgp import Plus
from geneticengine.grammars.sgp import Var
from geneticengine.metahandlers.floats import FloatList
from geneticengine.metahandlers.vars import VarRange
from geneticengine.metrics import f1_score

# An example of classification using Probabilistic GE (https://arxiv.org/pdf/2103.08389.pdf).
# The main differences with the normal classification example are the addition of weights/probabilities to the
# production rules of the grammar (lines 81-83), and adding the evolve_grammar parameter in the GP class (line 159).
# Notice that the weights don't need to be added to the grammar, as by default all production rules have the same
# weight/probability.

DATASET_NAME = "Banknote"
DATA_FILE_TRAIN = f"examples/data/{DATASET_NAME}/Train.csv"
DATA_FILE_TEST = f"examples/data/{DATASET_NAME}/Test.csv"

bunch = pd.read_csv(DATA_FILE_TRAIN, delimiter=" ")
target = bunch.y
data = bunch.drop(["y"], axis=1)

feature_names = list(data.columns.values)
feature_indices = {}
for i, n in enumerate(feature_names):
    feature_indices[n] = i

# Prepare Grammar
Var.__init__.__annotations__["name"] = Annotated[str, VarRange(feature_names)]
Var.feature_indices = feature_indices  # type: ignore


@dataclass
class Literal(Number):
    val: Annotated[float, FloatList([-1, -0.1, -0.01, -0.001, 1, 0.1, 0.01, 0.001])]

    def evaluate(self, **kwargs):
        return self.val

    def __str__(self) -> str:
        return str(self.val)


prods = [Plus, Mul, SafeDiv, Literal, Var]


def preprocess():
    return extract_grammar(
        prods,
        Number,
    )


for prod in prods:
    get_gengy(prod)
    prod.__dict__["__gengy__"]["weight"] = 1

X_train, X_test, y_train, y_test = train_test_split(
    data.values,
    target.values,
    test_size=0.75,
)


def fitness_function(n: Number):
    X = X_train
    y = y_train

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


def fitness_test_function(n: Number):
    X = X_test
    y = y_test

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
    representation="TreeBasedRepresentation",
    depth_aware=True,
):
    if representation == "ge":
        representation = GrammaticalEvolutionRepresentation
    elif representation == "sge":
        representation = StructuredGrammaticalEvolutionRepresentation
    elif representation == "dsge":
        representation = DynamicStructuredGrammaticalEvolutionRepresentation
    else:
        representation = TreeBasedRepresentation

    alg = SimpleGP(
        g,
        representation=representation,
        problem=SingleObjectiveProblem(
            minimize=False,
            fitness_function=fitness_function,
            target_fitness=None,
        ),
        probability_crossover=1,
        evolve_grammar=PGECallback(),
        probability_mutation=0.5,
        number_of_generations=20,
        max_depth=10,
        max_init_depth=6,
        population_size=500,
        selection_method=("tournament", 2),
        n_elites=50,
        seed=seed,
        timer_stop_criteria=mode,
    )
    (b, bf, bp) = alg.evolve()
    return b, bf, g


if __name__ == "__main__":
    g = preprocess()
    print(g)
    b, bf, final_grammar = evolve(g, 123, False)
    print(b)
    print(f"With fitness: {bf}")
    print("Final grammar:")
    print(final_grammar)
