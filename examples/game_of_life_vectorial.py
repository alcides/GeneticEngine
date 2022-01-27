import sys
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

def prepare_data(DATASET_NAME):

    DATA_FILE_TRAIN = "examples/data/{}/Train.csv".format(DATASET_NAME)
    DATA_FILE_TEST = "examples/data/{}/Test.csv".format(DATASET_NAME)

    train = np.genfromtxt(DATA_FILE_TRAIN, skip_header=1, delimiter=",", dtype=int)
    Xtrain = train[:, :-1]
    Xtrain = Xtrain.reshape(train.shape[0], MATRIX_ROW_SIZE, MATRIX_COL_SIZE)
    ytrain =train[:, -1]

    test = np.genfromtxt(DATA_FILE_TEST, skip_header=1, delimiter=",", dtype=int)
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

    def summing(self,matrix):
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
    elif isinstance(e, SumAll):
        return lambda line: sum(sum(line))
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




def preprocess(output_folder,method):
    '''
        Options for methor are [standard], [row], [col], [row_col], [cube], [row_col_cube], [sum_all].
    '''
    if method == 'standard':
        grammar = extract_grammar([And, Or, Not, MatrixElement], Condition)
    elif method == 'row':
        grammar = extract_grammar([And, Or, Not, MatrixElement, MatrixElementsRow, MatrixSum, Equals, GreaterThan, LessThan, Literal], Condition)
    elif method == 'col':
        grammar = extract_grammar([And, Or, Not, MatrixElement, MatrixElementsCol, MatrixSum, Equals, GreaterThan, LessThan, Literal], Condition)
    elif method == 'row_col':
        grammar = extract_grammar([And, Or, Not, MatrixElement, MatrixElementsRow, MatrixElementsCol, MatrixSum, Equals, GreaterThan, LessThan, Literal], Condition)
    elif method == 'cube':
        grammar = extract_grammar([And, Or, Not, MatrixElement, MatrixElementsCube, MatrixSum, Equals, GreaterThan, LessThan, Literal], Condition)
    elif method == 'row_col_cube':
        grammar = extract_grammar([And, Or, Not, MatrixElement, MatrixElementsRow, MatrixElementsCol, MatrixElementsCube, MatrixSum, Equals, GreaterThan, LessThan, Literal], Condition)
    elif method == 'sum_all':
        grammar = extract_grammar([And, Or, Not, MatrixElement, SumAll, Equals, GreaterThan, LessThan, Literal], Condition)
    else:
        raise NotImplementedError(f'Method ({method}) not implemented! Choose from: [standard], [row], [col], [row_col], [cube], [row_col_cube], [sum_all]')

    file1 = open(f"results/csvs/{output_folder}/grammar.txt","w")
    file1.write(str(grammar))
    file1.close()
    
    print(grammar)
    return grammar

def evolve(fitness_function, output_folder, g, seed, mode):
    alg = GP(
        g,
        fitness_function,
        representation=treebased_representation,
        number_of_generations=150,
        population_size=100,
        max_depth=15,
        favor_less_deep_trees=True,
        probability_crossover=0.75,
        probability_mutation=0.01,
        selection_method=("tournament", 2),
        minimize=False,
        seed=seed,
        timer_stop_criteria=mode,
        safe_gen_to_csv=(f'{output_folder}/run_seed={seed}','all'),
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
    args = sys.argv
    print(args)
    output_folder = args[1] # 'GoL/grammar_col'
    method = args[2] # 'col'
    dataset_name = args[3] # 'GameOfLife'

    folder = f'./results/csvs/{output_folder}'
    # import IPython as ip
    # ip.embed()
    if not os.path.isdir(folder):
        os.mkdir(folder)

    g = preprocess(output_folder,method)
    
    Xtrain, Xtest, ytrain, ytest = prepare_data(dataset_name)    

    def fitness_function(i: Condition):
        _clf = evaluate(i)
        ypred = [_clf(line) for line in np.rollaxis(Xtrain, 0)]
        return f1_score(ytrain, ypred)
    
    for i in range(30):
        evolve(fitness_function, output_folder, g, i, False)