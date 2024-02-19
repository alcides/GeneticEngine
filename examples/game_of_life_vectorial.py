from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Annotated
from typing import Any
from typing import Callable

import numpy as np
from sklearn.metrics import f1_score

from geml.simplegp import SimpleGP
from geneticengine.grammar.decorators import abstract
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.grammar import Grammar
from geneticengine.problems import Problem
from geneticengine.problems import SingleObjectiveProblem
from geml.grammars.coding.classes import Condition
from geml.grammars.coding.classes import Expr
from geml.grammars.coding.classes import Number
from geml.grammars.coding.conditions import Equals
from geml.grammars.coding.conditions import GreaterThan
from geml.grammars.coding.conditions import LessThan
from geml.grammars.coding.logical_ops import And
from geml.grammars.coding.logical_ops import Not
from geml.grammars.coding.logical_ops import Or
from geml.grammars.coding.numbers import Literal
from geneticengine.grammar.metahandlers.ints import IntRange

# ===================================
# This is an example on how to use GeneticEngine to solve a GP problem.
# We define the tree structure of the representation and then we define the fitness function for our problem
# In this example we are Reversing Game of Life Using GP
# We used the GameOfLife dataset stored in examples/data folder.
# This example differs from the normal game_of_life.py through the addition of Vectorial-GP-style grammar (https://link.springer.com/chapter/10.1007/978-3-030-16670-0_14).
# ===================================

MATRIX_ROW_SIZE = 3
MATRIX_COL_SIZE = 3


def prepare_data(DATASET_NAME):
    DATA_FILE_TRAIN = f"examples/data/{DATASET_NAME}/Train.csv"
    DATA_FILE_TEST = f"examples/data/{DATASET_NAME}/Test.csv"

    train = np.genfromtxt(
        DATA_FILE_TRAIN,
        skip_header=1,
        delimiter=",",
        dtype=int,
    )
    Xtrain = train[:, :-1]
    Xtrain = Xtrain.reshape(train.shape[0], MATRIX_ROW_SIZE, MATRIX_COL_SIZE)
    ytrain = train[:, -1]

    test = np.genfromtxt(
        DATA_FILE_TEST,
        skip_header=1,
        delimiter=",",
        dtype=int,
    )
    Xtest = test[:, :-1]
    Xtest = Xtest.reshape(test.shape[0], MATRIX_ROW_SIZE, MATRIX_COL_SIZE)
    ytest = test[:, -1]

    return Xtrain, Xtest, ytrain, ytest


@dataclass
class MatrixElement(Condition):
    row: Annotated[int, IntRange(0, MATRIX_ROW_SIZE - 1)]
    column: Annotated[int, IntRange(0, MATRIX_COL_SIZE - 1)]

    def __str__(self) -> str:
        return f"(X[{self.row}, {self.column}])"


@abstract
class Matrix(ABC):
    pass


@dataclass
class MatrixElementsRow(Matrix):
    row: Annotated[int, IntRange(0, MATRIX_ROW_SIZE - 1)]
    col1: Annotated[int, IntRange(0, MATRIX_COL_SIZE)]
    col2: Annotated[int, IntRange(0, MATRIX_COL_SIZE)]

    def __str__(self) -> str:
        return f"X[{self.row}, {self.col1} : {self.col2}]"


@dataclass
class MatrixElementsCol(Matrix):
    row1: Annotated[int, IntRange(0, MATRIX_ROW_SIZE)]
    row2: Annotated[int, IntRange(0, MATRIX_ROW_SIZE)]
    col: Annotated[int, IntRange(0, MATRIX_COL_SIZE - 1)]

    def __str__(self) -> str:
        return f"X[{self.row1} : {self.row2}, {self.col}]"


@dataclass
class MatrixElementsCube(Matrix):
    row1: Annotated[int, IntRange(0, MATRIX_ROW_SIZE)]
    row2: Annotated[int, IntRange(0, MATRIX_ROW_SIZE)]
    col1: Annotated[int, IntRange(0, MATRIX_COL_SIZE)]
    col2: Annotated[int, IntRange(0, MATRIX_COL_SIZE)]

    def __str__(self) -> str:
        return f"X[{self.row1} : {self.row2}, {self.col1} : {self.col2}]"


@dataclass
class MatrixSum(Number):
    matrix: Matrix

    def summing(self, matrix):
        s = sum(matrix)
        if isinstance(s, int) or isinstance(s, np.int32) or isinstance(s, np.int64):
            return s
        else:
            return sum(s)

    def __str__(self) -> str:
        return f"(sum({self.matrix}))"


@dataclass
class SumAll(Number):
    def __str__(self) -> str:
        return "(sum(X))"


def flat_sum(s):
    if isinstance(s, int):
        return s
    elif isinstance(s, float):
        return s
    else:
        return sum(s)


def evaluate(e: Expr | Matrix | Number) -> Callable[[Any], float]:
    if isinstance(e, And):
        l = e.left
        r = e.right
        return lambda line: evaluate(l)(line) and evaluate(r)(line)
    elif isinstance(e, Or):
        l = e.left
        r = e.right
        return lambda line: evaluate(l)(line) or evaluate(r)(line)
    elif isinstance(e, Not):
        cond = e.cond
        return lambda line: not evaluate(cond)(line)
    elif isinstance(e, MatrixElement):
        row = e.row
        col = e.column
        return lambda line: line[row, col]
    elif isinstance(e, MatrixElementsRow):
        row = e.row
        c1 = e.col1
        c2 = e.col2
        if e.col1 <= e.col2:
            return lambda line: line[row, c1:c2]
        else:
            return lambda line: line[row, c2:c1]
    elif isinstance(e, MatrixElementsCol):
        row1 = e.row1
        row2 = e.row2
        col = e.col
        if e.row1 <= e.row2:
            return lambda line: line[row1:row2, col]
        else:
            return lambda line: line[row2:row1, col]
    elif isinstance(e, MatrixElementsCube):
        row1 = e.row1
        row2 = e.row2
        col1 = e.col1
        col2 = e.col2
        if e.row1 <= e.row2:
            if e.col1 <= e.col2:
                return lambda line: line[row1:row2, col1:col2]
            else:
                return lambda line: line[row1:row2, col2:col1]
        else:
            if e.col1 <= e.col2:
                return lambda line: line[row2:row1, col1:col2]
            else:
                return lambda line: line[row2:row1, col2:col1]
    elif isinstance(e, MatrixSum):
        m = e.matrix
        s = e.summing
        return lambda line: s(evaluate(m)(line))
    elif isinstance(e, SumAll):
        return lambda line: flat_sum(line)
    elif isinstance(e, Literal):
        v = e.val
        return lambda _: v
    elif isinstance(e, Equals):
        lneq: Number = e.left
        rneq: Number = e.right
        return lambda line: evaluate(lneq)(line) == evaluate(rneq)(line)
    elif isinstance(e, GreaterThan):
        lngt: Number = e.left
        rngt: Number = e.right
        return lambda line: evaluate(lngt)(line) > evaluate(rngt)(line)
    elif isinstance(e, LessThan):
        lnlt: Number = e.left
        rnlt: Number = e.right
        return lambda line: evaluate(lnlt)(line) < evaluate(rnlt)(line)
    else:
        raise NotImplementedError(str(e))


grammars = {
    "standard": extract_grammar([And, Or, Not, MatrixElement], Condition),
    "row": extract_grammar(
        [
            And,
            Or,
            Not,
            MatrixElement,
            MatrixElementsRow,
            MatrixSum,
            Equals,
            GreaterThan,
            LessThan,
            Literal,
        ],
        Condition,
    ),
    "col": extract_grammar(
        [
            And,
            Or,
            Not,
            MatrixElement,
            MatrixElementsCol,
            MatrixSum,
            Equals,
            GreaterThan,
            LessThan,
            Literal,
        ],
        Condition,
    ),
    "row_col": extract_grammar(
        [
            And,
            Or,
            Not,
            MatrixElement,
            MatrixElementsRow,
            MatrixElementsCol,
            MatrixSum,
            Equals,
            GreaterThan,
            LessThan,
            Literal,
        ],
        Condition,
    ),
    "cube": extract_grammar(
        [
            And,
            Or,
            Not,
            MatrixElement,
            MatrixElementsCube,
            MatrixSum,
            Equals,
            GreaterThan,
            LessThan,
            Literal,
        ],
        Condition,
    ),
    "row_col_cube": extract_grammar(
        [
            And,
            Or,
            Not,
            MatrixElement,
            MatrixElementsRow,
            MatrixElementsCol,
            MatrixElementsCube,
            MatrixSum,
            Equals,
            GreaterThan,
            LessThan,
            Literal,
        ],
        Condition,
    ),
    "sum_all": extract_grammar(
        [
            And,
            Or,
            Not,
            MatrixElement,
            SumAll,
            Equals,
            GreaterThan,
            LessThan,
            Literal,
        ],
        Condition,
    ),
}


dataset_name = "GameOfLife"
Xtrain, Xtest, ytrain, ytest = prepare_data(dataset_name)


def fitness_function(i: Condition):
    _clf = evaluate(i)
    ypred = [_clf(line) for line in np.rollaxis(Xtrain, 0)]
    return f1_score(ytrain, ypred)


class GameOfLifeVectorialBenchmark:
    def __init__(self, method: str = "standard"):
        self.grammar = grammars[method]

    def get_problem(self) -> Problem:
        return SingleObjectiveProblem(
            minimize=False,
            fitness_function=fitness_function,
        )

    def get_grammar(self) -> Grammar:
        return self.grammar

    def main(self, **args):
        g = self.get_grammar()

        alg = SimpleGP(
            grammar=g,
            minimize=True,
            fitness_function=fitness_function,
            max_evaluations=10000,
            population_size=50,
            max_depth=10,
            # favor_less_complex_trees=True,
            # crossover_probability=0.75,
            # mutation_probability=0.01,
            # selection_method=("tournament", 2),
            **args,
        )
        best = alg.search()
        print(
            f"Fitness of {best.get_fitness(alg.get_problem())} by genotype: {best.genotype} with phenotype: {best.get_phenotype()}",
        )

        _clf = evaluate(best.get_phenotype())
        ypred = [_clf(line) for line in np.rollaxis(Xtest, 0)]
        print("GeneticEngine Test F1 score:", f1_score(ytest, ypred))


if __name__ == "__main__":
    GameOfLifeVectorialBenchmark(method="col").main(seed=0)
