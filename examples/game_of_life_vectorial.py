import os
from abc import ABC
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Tuple
import numpy as np
from sklearn.metrics import f1_score
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.decorators import abstract
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.metahandlers.ints import IntRange
from geneticengine.algorithms.gp.gp import GP

from geneticengine.grammars.coding.logical_ops import And, Or, Not
from geneticengine.grammars.coding.conditions import Equals, GreaterThan, LessThan
from geneticengine.grammars.coding.classes import Expr, Condition, Number
from geneticengine.grammars.coding.numbers import Literal

MATRIX_ROW_SIZE = 3
MATRIX_COL_SIZE = 3


DATASET_NAME = "GameOfLifeVectorial"
DATA_FILE_TRAIN = "examples/data/{}/Train.csv".format(DATASET_NAME)
DATA_FILE_TEST = "examples/data/{}/Test.csv".format(DATASET_NAME)

train = np.genfromtxt(DATA_FILE_TRAIN, skip_header=1, delimiter=",", dtype=bool)
Xtrain = train[:, :-1]
Xtrain = Xtrain.reshape(train.shape[0], MATRIX_ROW_SIZE, MATRIX_COL_SIZE)
ytrain =train[:, -1] 

test = np.genfromtxt(DATA_FILE_TEST, skip_header=1, delimiter=",", dtype=bool)
Xtest = test[:, :-1]
Xtest = Xtest.reshape(test.shape[0], MATRIX_ROW_SIZE, MATRIX_COL_SIZE)
ytest = test[:, -1] 



@dataclass
class MatrixElement(Condition):
    row: Annotated[int, IntRange(0, MATRIX_ROW_SIZE - 1)]
    column: Annotated[int, IntRange(0, MATRIX_COL_SIZE - 1)]

    def __str__(self) -> str:
        return f"(X[{self.row}, {self.column}])"

@abstract
class Array(ABC):
    pass

@abstract
class Matrix(ABC):
    pass

@dataclass
class MatrixElementsRow(Array):
    row: Annotated[int, IntRange(0, MATRIX_ROW_SIZE - 1)]
    col1: Annotated[int, IntRange(0, MATRIX_COL_SIZE)]
    col2: Annotated[int, IntRange(0, MATRIX_COL_SIZE)]

    def __str__(self) -> str:
        return f"X[{self.row}, {self.col1} : {self.col2}]"

@dataclass
class MatrixElementsCol(Array):
    row1: Annotated[int, IntRange(0, MATRIX_ROW_SIZE)]
    row2: Annotated[int, IntRange(0, MATRIX_ROW_SIZE)]
    col: Annotated[int, IntRange(0, MATRIX_COL_SIZE - 1)]

    def __str__(self) -> str:
        return f"X[{self.row1} : {self.row2}, {self.col}]"

@dataclass
class ArraySum(Number):
    array: Array
    
    def __str__(self) -> str:
        return f"(sum({self.array}))"

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

    def summing(self,matrix):
        s = sum(matrix)
        if type(s) == int:
            return s
        else:
            return sum(s)
    
    def __str__(self) -> str:
        return f"(sum({self.matrix}))"


def evaluate(e: Expr) -> Callable[[Any], float]:

    if isinstance(e, And):
        return lambda line: evaluate(e.left)(line) and evaluate(e.right)(line)
    elif isinstance(e, Or):
        return lambda line: evaluate(e.left)(line) or evaluate(e.right)(line)
    elif isinstance(e, Not):
        return lambda line: not evaluate(e.cond)(line)
    elif isinstance(e, MatrixElement):
        return lambda line: line[e.row, e.column]
    elif isinstance(e, MatrixElementsRow):
        if e.col1 <= e.col2:
            return lambda line: line[e.row, e.col1 : e.col2]
        else:
            return lambda line: line[e.row, e.col2 : e.col1]
    elif isinstance(e, MatrixElementsCol):
        if e.row1 <= e.row2:
            return lambda line: line[e.row1 : e.row2, e.col]
        else:
            return lambda line: line[e.row2 : e.row1, e.col]
    elif isinstance(e, ArraySum):
        return lambda line: sum(evaluate(e.array)(line))
    elif isinstance(e, MatrixElementsCube):
        if e.row1 <= e.row2:
            if e.col1 <= e.col2:
                return lambda line: line[e.row1 : e.row2, e.col1 : e.col2]
            else:
                return lambda line: line[e.row1 : e.row2, e.col2 : e.col1]
        else:
            if e.col1 <= e.col2:
                return lambda line: line[e.row2 : e.row1, e.col1 : e.col2]
            else:
                return lambda line: line[e.row2 : e.row1, e.col2 : e.col1]
    elif isinstance(e, MatrixSum):
        return lambda line: e.summing(evaluate(e.matrix)(line))
    elif isinstance(e, Equals):
        return lambda line: evaluate(e.left)(line) == evaluate(e.right)(line)
    elif isinstance(e, GreaterThan):
        return lambda line: evaluate(e.left)(line) > evaluate(e.right)(line)
    elif isinstance(e, LessThan):
        return lambda line: evaluate(e.left)(line) < evaluate(e.right)(line)
    elif isinstance(e, Literal):
        return lambda _: e.val
    else:
        raise NotImplementedError(str(e))


def fitness_function(i: Condition):
    _clf = evaluate(i)
    ypred = [_clf(line) for line in np.rollaxis(Xtrain, 0)]
    return f1_score(ytrain, ypred)


def preprocess():
    # grammar = extract_grammar([And, Or, Not, MatrixElement, MatrixElementsRow, MatrixElementsCol, ArraySum, MatrixElementsCube, MatrixSum, Equals, GreaterThan, LessThan, Literal], Condition)
    grammar = extract_grammar([And, Or, Not, MatrixElement, MatrixElementsRow, MatrixElementsCol, ArraySum, Equals, GreaterThan, LessThan, Literal], Condition) # Finds solution!!!
    print(grammar)
    return grammar

def evolve(g, seed, mode):
    alg = GP(
        g,
        fitness_function,
        representation=treebased_representation,
        number_of_generations=50,
        population_size=100,
        max_depth=15,
        favor_less_deep_trees=True,
        probability_crossover=0.75,
        probability_mutation=0.01,
        selection_method=("tournament", 2),
        minimize=False,
        seed=seed,
        timer_stop_criteria=mode,
        safe_gen_to_csv=(f'GoL_vectorial_(seed={seed})',False),
    )
    (b, bf, bp) = alg.evolve(verbose=1)

    print("Best individual:", bp)
    print("Genetic Engine Train F1 score:", bf)
    
    _clf = evaluate(bp)
    ypred = [_clf(line) for line in np.rollaxis(Xtest, 0)]
    print("GeneticEngine Test F1 score:", f1_score(ytest, ypred))

    return b, bf


# import IPython as ip
# ip.embed()

if __name__ == "__main__":
 
    # # # Generate dataset
    # # Train
    # Xtrain, ytrain = generate_dataset(1000)
    # Xtest, ytest = generate_dataset(1000)
    # _x = Xtrain.reshape(1000, 9)
    # _y = ytrain.reshape(1000, 1)
    # np.savetxt("Train.csv", np.concatenate([_x, _y], axis=1), fmt='%i', delimiter=",")

    # # Test
    # _x = Xtest.reshape(1000, 9)
    # _y = ytest.reshape(1000, 1)
    # np.savetxt("Test.csv", np.concatenate([_x, _y], axis=1), fmt='%i', delimiter=",")
    
    for i in range(30):
        g = preprocess()
        evolve(g, i, False)