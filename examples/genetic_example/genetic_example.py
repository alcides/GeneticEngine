from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Annotated

import numpy as np
import pandas as pd
from sklearn import metrics

from geml.simplegp import SimpleGP
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.grammar.metahandlers.vars import VarRange

dataset2 = pd.read_csv("dataset_secundario.csv", sep=";")


class DecisionNode(ABC):
    def evaluate(self, X): ...


class IntNode(ABC):
    def evaluate(self, X): ...


@dataclass
class Variable(IntNode):
    column_index: Annotated[int, IntRange(0, 2)]

    def evaluate(self, X):
        return X[:, self.column_index]

    def __str__(self):
        return f"var[{self.column_index}]"


@dataclass
class TotalCount(IntNode):
    col_dataset1: Annotated[int, IntRange(0, 2)]
    col_dataset2: Annotated[str, VarRange(["a", "b", "c"])]

    def count_values(self, i: int):
        return np.sum([1 for e in dataset2[self.col_dataset2] if e == i])

    def evaluate(self, X):
        return np.array([self.count_values(i) for i in X[:, self.col_dataset1]])

    def __str__(self):
        return f"sum(dataset_secundario[{self.col_dataset2} == dataset_primario[{self.col_dataset1}]])"


@dataclass
class And(DecisionNode):
    left: DecisionNode
    right: DecisionNode

    def evaluate(self, X):
        le = self.left.evaluate(X)
        re = self.right.evaluate(X)
        return le & re

    def __str__(self):
        return f"({self.left}) + ({self.right})"


@dataclass
class LessThan(DecisionNode):
    left: IntNode
    right: Annotated[int, IntRange(0, 100)]

    def evaluate(self, X):
        arrb = self.left.evaluate(X) < self.right
        arri = np.array(arrb, dtype=int)
        return arri

    def __str__(self):
        return f"({self.left}) < ({self.right})"


def main():
    dataset1 = pd.read_csv("dataset_principal.csv", sep=";")

    y = dataset1["y"].values
    X = dataset1.drop("y", axis=1).values

    def fitness_function(d: DecisionNode):
        y_pred = d.evaluate(X)
        return metrics.accuracy_score(y, y_pred)

    g = extract_grammar([Variable, And, LessThan, TotalCount], DecisionNode)

    alg = SimpleGP(
        grammar=g,
        fitness_function=fitness_function,
        representation="treebased",
        minimize=False,
        seed=122,
        population_size=10,
        max_evaluations=10000,
    )
    ind = alg.search()[0]
    print(ind)


main()
