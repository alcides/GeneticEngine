import os
from abc import ABC
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Tuple
import numpy as np
from sklearn.metrics import f1_score
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.treebased import treebased_representation
from geneticengine.metahandlers.ints import IntRange
from geneticengine.algorithms.gp.gp import GP

from geneticengine.grammars.coding.logical_ops import And, Or, Not
from geneticengine.grammars.coding.conditions import Equals
from geneticengine.grammars.coding.classes import Condition, Number
from geneticengine.grammars.coding.numbers import Literal


DATASET_NAME = "GameOfLifeVectorial"
DATA_FILE_TRAIN = "examples/data/{}/Train.csv".format(DATASET_NAME)
DATA_FILE_TEST = "examples/data/{}/Test.csv".format(DATASET_NAME)

train = np.genfromtxt(DATA_FILE_TRAIN, skip_header=1, delimiter=",", dtype=bool)
Xtrain = train[:, :-1]
Xtrain = Xtrain.reshape(train.shape[0], 3, 3)
ytrain =train[:, -1] 

test = np.genfromtxt(DATA_FILE_TEST, skip_header=1, delimiter=",", dtype=bool)
Xtest = test[:, :-1]
Xtest = Xtest.reshape(test.shape[0], 3, 3)
ytest = test[:, -1] 



class Expr(ABC):
    pass


@dataclass
class MatrixElement(Condition):
    row: Annotated[int, IntRange(0, 2)]
    column: Annotated[int, IntRange(0, 2)]

    def __str__(self) -> str:
        return f"(X[{self.row}, {self.column}])"

@dataclass
class MatrixSum(Number):
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
    elif isinstance(e, MatrixSum):
        return lambda line: sum(sum(line))
    elif isinstance(e, Equals):
        return lambda line: evaluate(e.left)(line) == evaluate(e.right)(line)
    elif isinstance(e, Literal):
        return lambda _: e.val
    else:
        raise NotImplementedError(str(e))


def fitness_function(i: Condition):
    _clf = evaluate(i)
    ypred = [_clf(line) for line in np.rollaxis(Xtrain, 0)]
    return f1_score(ytrain, ypred)


def preprocess():
    grammar = extract_grammar([And, Or, Not, MatrixElement, MatrixSum, Equals, Literal], Condition)
    print(grammar)
    return grammar

def evolve(g, seed, mode):
    alg = GP(
        g,
        treebased_representation,
        fitness_function,
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
    )
    (b, bf, bp) = alg.evolve(verbose=0)

    print("Best individual:", bp)
    print("Genetic Engine Train F1 score:", bf)
    
    _clf = evaluate(bp)
    ypred = [_clf(line) for line in np.rollaxis(Xtest, 0)]
    print("GeneticEngine Test F1 score:", f1_score(ytest, ypred))

    return b, bf

ind1 = And(MatrixElement(1,1),Or(Equals(MatrixSum(),Literal(3)),Equals(MatrixSum(),Literal(4))))
ind2 = And(Not(MatrixElement(1,1)),Equals(MatrixSum(),Literal(3)))
best_ind = Or(ind1,ind2)

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
    
    g = preprocess()
    evolve(g, 1, False)