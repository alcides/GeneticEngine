from __future__ import annotations

from dataclasses import dataclass
from math import isinf
from typing import Annotated

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from geml.simplegp import SimpleGP
from geneticengine.grammar.decorators import get_gengy
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.grammar import Grammar
from geneticengine.problems import Problem
from geneticengine.problems import SingleObjectiveProblem
from geml.grammars.basic_math import SafeDiv
from geml.grammars.sgp import Mul
from geml.grammars.sgp import Number
from geml.grammars.sgp import Plus
from geml.grammars.sgp import Var
from geneticengine.grammar.metahandlers.floats import FloatList
from geneticengine.grammar.metahandlers.vars import VarRange
from sklearn.metrics import f1_score

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


prods: list[type] = [Plus, Mul, SafeDiv, Literal, Var]


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


class ClassificationBenchmark:
    def get_problem(self) -> Problem:
        return SingleObjectiveProblem(minimize=False, fitness_function=fitness_function)

    def get_grammar(self) -> Grammar:
        return extract_grammar(
            prods,
            Number,
        )

    def main(self, **args):
        g = self.get_grammar()
        alg = SimpleGP(
            grammar=g,
            minimize=True,
            fitness_function=fitness_function,
            crossover_probability=1,
            mutation_probability=0.5,
            max_evaluations=10000,
            max_depth=10,
            population_size=50,
            selection_method=("tournament", 2),
            elitism=5,
            **args,
        )
        best = alg.search()
        print(
            f"Fitness of {best.get_fitness(alg.get_problem())} by genotype: {best.genotype} with phenotype: {best.get_phenotype()}",
        )


if __name__ == "__main__":
    ClassificationBenchmark().main(seed=0)
