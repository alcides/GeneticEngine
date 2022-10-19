from __future__ import annotations

import csv
import os
import sys
import time
from abc import ABC
from dataclasses import dataclass
from optparse import OptionParser
from typing import Annotated
from typing import Any
from typing import Callable
from typing import Tuple
from typing import Union

import global_vars as gv
import numpy as np
from sklearn.metrics import f1_score

from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.decorators import abstract
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.problems import SingleObjectiveProblem
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
from geneticengine.grammars.coding.classes import Condition
from geneticengine.grammars.coding.classes import Expr
from geneticengine.grammars.coding.classes import Number
from geneticengine.grammars.coding.conditions import Equals
from geneticengine.grammars.coding.conditions import GreaterThan
from geneticengine.grammars.coding.conditions import LessThan
from geneticengine.grammars.coding.logical_ops import And
from geneticengine.grammars.coding.logical_ops import Not
from geneticengine.grammars.coding.logical_ops import Or
from geneticengine.grammars.coding.numbers import Literal
from geneticengine.metahandlers.ints import IntRange

MATRIX_ROW_SIZE = 3
MATRIX_COL_SIZE = 3


def prepare_data(DATASET_NAME):

    DATA_FILE_TRAIN = f"GeneticEngine/examples/data/{DATASET_NAME}/Train.csv"
    DATA_FILE_TEST = f"GeneticEngine/examples/data/{DATASET_NAME}/Test.csv"

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
        if type(s) == int or type(s) == np.int32 or type(s) == np.int64:
            return s
        else:
            return sum(s)

    def __str__(self) -> str:
        return f"(sum({self.matrix}))"


@dataclass
class SumAll(Number):
    def __str__(self) -> str:
        return f"(sum(X))"


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
    elif isinstance(e, Equals):
        ln: Number = e.left
        rn: Number = e.right
        return lambda line: evaluate(ln)(line) == evaluate(rn)(line)
    elif isinstance(e, GreaterThan):
        ln: Number = e.left
        rn: Number = e.right
        return lambda line: evaluate(ln)(line) > evaluate(rn)(line)
    elif isinstance(e, LessThan):
        ln: Number = e.left
        rn: Number = e.right
        return lambda line: evaluate(ln)(line) < evaluate(rn)(line)
    elif isinstance(e, Literal):
        v = e.val
        return lambda _: v
    else:
        raise NotImplementedError(str(e))


def preprocess(method):
    """
    Options for methor are [standard], [row], [col], [row_col], [cube], [row_col_cube], [sum_all].
    """
    if method == "standard":
        grammar = extract_grammar([And, Or, Not, MatrixElement], Condition)
    elif method == "row":
        grammar = extract_grammar(
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
        )
    elif method == "col":
        grammar = extract_grammar(
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
        )
    elif method == "row_col":
        grammar = extract_grammar(
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
        )
    elif method == "cube":
        grammar = extract_grammar(
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
        )
    elif method == "row_col_cube":
        grammar = extract_grammar(
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
    elif method == "sum_all":
        grammar = extract_grammar(
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
        )
    else:
        raise NotImplementedError(
            f"Method ({method}) not implemented! Choose from: [standard], [row], [col], [row_col], [cube], [row_col_cube], [sum_all]",
        )

    print(grammar)
    return grammar


Xtrain, Xtest, ytrain, ytest = prepare_data("GameOfLife")


def fitness_function(i: Condition):
    _clf = evaluate(i)
    ypred = [_clf(line) for line in np.rollaxis(Xtrain, 0)]
    return f1_score(ytrain, ypred)


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

    g = preprocess("row_col_cube")
    alg = GP(
        g,
        representation=representation,
        problem=SingleObjectiveProblem(
            minimize=False,
            fitness_function=fitness_function,
            target_fitness=None,
        ),
        probability_crossover=gv.PROBABILITY_CROSSOVER,
        probability_mutation=gv.PROBABILITY_MUTATION,
        number_of_generations=gv.NUMBER_OF_GENERATIONS,
        max_depth=gv.MAX_DEPTH,
        population_size=gv.POPULATION_SIZE,
        selection_method=gv.SELECTION_METHOD,
        n_elites=gv.N_ELITES,
        seed=seed,
        timer_stop_criteria=mode,
        save_to_csv=save_to_csv,
        save_genotype_as_string=False,
    )
    (b, bf, bp) = alg.evolve(verbose=1)
    return b, bf


if __name__ == "__main__":
    representations = ["treebased_representation", "ge", "dsge"]

    parser = OptionParser()
    parser.add_option("-s", "--seed", dest="seed", type="int")
    parser.add_option("-r", "--representation", dest="representation", type="int")
    parser.add_option(
        "-t",
        "--timed",
        dest="timed",
        action="store_const",
        const=True,
        default=False,
    )
    (options, args) = parser.parse_args()

    timed = options.timed
    seed = options.seed
    example_name = __file__.split(".")[0].split("\\")[-1].split("/")[-1]
    representation = representations[options.representation]
    print(seed, example_name, representation)
    evol_method = evolve

    mode = "generations"
    if timed:
        mode = "time"
    dest_file = f"{gv.RESULTS_FOLDER}/{mode}/{example_name}/{representation}/{seed}.csv"

    start = time.time()
    b, bf = evolve(seed, timed, dest_file, representation)
    end = time.time()
    csv_row = [mode, example_name, representation, seed, bf, (end - start), b]
    with open(
        f"./{gv.RESULTS_FOLDER}/{mode}/{example_name}/{representation}/main.csv",
        "a",
        newline="",
    ) as outfile:
        writer = csv.writer(outfile)
        writer.writerow(csv_row)
