from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
from typing import Annotated
from typing import Any
from typing import Callable

import numpy as np
from sklearn.metrics import f1_score

from examples.benchmarks.benchmark import Benchmark, example_run
from examples.benchmarks.datasets import get_game_of_life
from geml.grammars.coding.conditions import Equals, GreaterThan, LessThan
from geml.grammars.coding.numbers import Literal, Number
from geneticengine.grammar.grammar import extract_grammar, Grammar
from geneticengine.problems import Problem, SingleObjectiveProblem
from geml.grammars.coding.classes import Condition
from geml.grammars.coding.classes import Expr
from geml.grammars.coding.logical_ops import And
from geml.grammars.coding.logical_ops import Not
from geml.grammars.coding.logical_ops import Or
from geneticengine.grammar.metahandlers.ints import IntRange

# [Include all the necessary imports and class definitions here]


MATRIX_ROW_SIZE = 3
MATRIX_COL_SIZE = 3


@dataclass
class MatrixElement(Condition):
    row: Annotated[int, IntRange(0, MATRIX_ROW_SIZE - 1)]
    column: Annotated[int, IntRange(0, MATRIX_COL_SIZE - 1)]

    def __str__(self) -> str:
        return f"(X[{self.row}, {self.column}])"


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


class GameOfLifeVectorialBenchmark(Benchmark):
    def __init__(self, X, y):
        self.setup_problem(X, y)
        self.setup_grammar()

    def setup_problem(self, X, y):
        def fitness_function(i: Condition):
            _clf = evaluate(i)
            ypred = [_clf(line) for line in np.rollaxis(X, 0)]
            return f1_score(y, ypred)

        self.problem = SingleObjectiveProblem(minimize=False, fitness_function=fitness_function, target=1)

    def setup_grammar(self):
        self.grammar = extract_grammar(
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
        )

    def get_problem(self) -> Problem:
        return self.problem

    def get_grammar(self) -> Grammar:
        return self.grammar


if __name__ == "__main__":
    X, y = get_game_of_life()
    example_run(GameOfLifeVectorialBenchmark(X, y))
