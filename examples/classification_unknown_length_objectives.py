from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from geml.simplegp import SimpleGP
from geneticengine.grammar.decorators import abstract
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.grammar import Grammar
from geneticengine.problems import MultiObjectiveProblem
from geneticengine.problems import Problem
from geml.grammars.basic_math import SafeDiv
from geml.grammars.basic_math import SafeLog
from geml.grammars.basic_math import SafeSqrt
from geml.grammars.sgp import Mul
from geml.grammars.sgp import Number
from geml.grammars.sgp import Plus
from geml.grammars.sgp import Var
from geneticengine.grammar.metahandlers.floats import FloatList
from geneticengine.grammar.metahandlers.vars import VarRange

# ===================================
# This is an example of normal classification using normal GP,
# with a lexicase selection algorithm as the parent selection.
# We used the Banknote dataset stored in examples/data folder
# ===================================

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
    Literal,
    MinusOne,
    MinusPointOne,
    MinusPointtOne,
    MinusPointttOne,
    One,
    PointOne,
    PointtOne,
    PointttOne,
]


@dataclass
class Literal2(Number):
    val: Annotated[float, FloatList([-1, -0.1, -0.01, -0.001, 1, 0.1, 0.01, 0.001])]

    def evaluate(self, **kwargs):
        return self.val

    def __str__(self) -> str:
        return str(self.val)


X_train, X_test, y_train, y_test = train_test_split(
    data.values,
    target.values,
    test_size=0.75,
)


def fitness_function_lexicase(n: Number):
    X = X_test
    y = y_test

    variables = {}
    for x in feature_names:
        i = feature_indices[x]
        variables[x] = X[:, i]

    y_pred = n.evaluate(**variables)
    if type(y_pred) in [np.float64, int, float] or y_pred.shape != y.shape:
        """If n does not use variables, the output will be scalar."""
        y_pred = np.full(len(y), y_pred)

    return [int(p == r) for (p, r) in zip(y, y_pred)]


class ClassificationLexicaseBenchmark:
    def get_problem(self) -> Problem:
        minimizelist = False

        def single_criteria_test(n: Number) -> float:
            fitnesses = fitness_function_lexicase(n)
            return sum(-fit if minimizelist else fit for fit in fitnesses)

        return MultiObjectiveProblem(
            minimize=minimizelist,
            fitness_function=fitness_function_lexicase,
            best_individual_criteria_function=single_criteria_test,
        )

    def get_grammar(self) -> Grammar:
        return extract_grammar(
            [Plus, Mul, SafeDiv, Literal2, Var, SafeSqrt, SafeLog],
            Number,
        )

    def main(self, **args):
        g = self.get_grammar()
        prob = self.get_problem()
        alg = SimpleGP(
            g,
            problem=prob,
            crossover_probability=0.75,
            mutation_probability=0.01,
            number_of_generations=50,
            max_depth=15,
            population_size=50,
            selection_method=("lexicase",),
            n_elites=5,
            **args,
        )
        best = alg.search()
        print(
            f"Fitness of {best.get_fitness(prob)} by genotype: {best.genotype} with phenotype: {best.get_phenotype()}",
        )


if __name__ == "__main__":
    ClassificationLexicaseBenchmark().main(seed=0)
