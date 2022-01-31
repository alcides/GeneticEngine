import os
from abc import ABC
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Tuple
import numpy as np
from sklearn.metrics import f1_score
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.metahandlers.ints import IntRange
from geneticengine.algorithms.gp.gp import GP

from geneticengine.grammars.coding.logical_ops import And, Or, Not
from geneticengine.grammars.coding.classes import Expr, Condition, Number

DATASET_NAME = "GameOfLife"
DATA_FILE_TRAIN = "examples/data/{}/Train.csv".format(DATASET_NAME)
DATA_FILE_TEST = "examples/data/{}/Test.csv".format(DATASET_NAME)
OUTPUT_FOLDER = 'GoL/grammar_standard'



train = np.genfromtxt(DATA_FILE_TRAIN, skip_header=1, delimiter=",")
Xtrain = train[:, :-1]
Xtrain = Xtrain.reshape(train.shape[0], 3, 3)
ytrain = train[:, -1]

test = np.genfromtxt(DATA_FILE_TEST, skip_header=1, delimiter=",")
Xtest = test[:, :-1]
Xtest = Xtest.reshape(test.shape[0], 3, 3)
ytest = test[:, -1]

@dataclass
class MatrixElement(Condition):
    row: Annotated[int, IntRange(0, 2)]
    column: Annotated[int, IntRange(0, 2)]

    def __str__(self) -> str:
        return f"(X[{self.row}, {self.column}])"


def evaluate(e: Expr) -> Callable[[Any], float]:

    if isinstance(e, And):
        f1 = evaluate(e.left)
        f2 = evaluate(e.right)
        return lambda line: f1(line) and f2(line)
    elif isinstance(e, Or):
        f1 = evaluate(e.left)
        f2 = evaluate(e.right)
        return lambda line: f1(line) or f2(line)
    elif isinstance(e, Not):
        f1 = evaluate(e.cond)
        return lambda line: not f1(line)
    elif isinstance(e, MatrixElement):
        r = e.row
        c = e.column
        return lambda line: line[r, c]
    else:
        raise NotImplementedError(str(e))


def fitness_function(i: Expr):
    _clf = evaluate(i)
    ypred = [_clf(line) for line in np.rollaxis(Xtrain, 0)]
    return f1_score(ytrain, ypred)

def preprocess():
    grammar = extract_grammar([And, Or, Not, MatrixElement], Condition)

    file1 = open(f"results/csvs/{OUTPUT_FOLDER}/grammar.txt","w")
    file1.write(str(grammar))
    file1.close()
    
    print(grammar)
    return grammar

def evolve(g, seed, mode, representation):
    alg = GP(
        g,
        fitness_function,
        representation=representation,
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
        safe_gen_to_csv=(f'{OUTPUT_FOLDER}/run_seed={seed}','all'),
    )
    (b, bf, bp) = alg.evolve(verbose=0)

    print("Best individual:", bp)
    print("Genetic Engine Train F1 score:", bf)

    _clf = evaluate(bp)
    ypred = [_clf(line) for line in np.rollaxis(Xtest, 0)]
    print("GeneticEngine Test F1 score:", f1_score(ytest, ypred))

    return b, bf



if __name__ == "__main__":
    g = preprocess()
    bf, b = evolve(g, 0, False, treebased_representation)
    print(b)
    print(f"With fitness: {bf}")