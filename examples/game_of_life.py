from abc import ABC
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Tuple
import numpy as np
from numpy.lib.arraysetops import isin
from scipy.sparse import data
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.treebased import treebased_representation
from geneticengine.metahandlers.ints import IntRange
from geneticengine.algorithms.gp.gp import GP


def game_of_life_rule(m):
    """Given a 3x3 matriz, outputs the correct result of Game of Life."""
    cell = m[1, 1]
    neighbours = np.sum(m) - cell
    if cell and neighbours in [2, 3]:
        return 1
    elif not cell and neighbours == 3:
        return 1
    else:
        return 0


def generate_dataset(n: int) -> Tuple[Any, Any]:
    """Generates a pair of Nx3x3 matrices of input boards, and the next value for the middle position of each board."""
    m = np.random.randint(0, 2, n * 9).reshape(n, 3, 3)
    r = np.fromiter((game_of_life_rule(xi) for xi in m), m.dtype)
    return (m, r)


def learn_predictor(Xtrain, ytrain, Xtest, ytest):

    clf = DecisionTreeClassifier(random_state=0, max_depth=5)
    clf.fit(Xtrain.reshape(1000, 9), ytrain)
    ypred = clf.predict(Xtest.reshape(1000, 9))
    accuracy = (ypred == ytest).mean()
    print("Decision Tree Test Accuracy:", accuracy)


class Expr(ABC):
    pass

@dataclass
class And(Expr):
    e1: Expr
    e2: Expr

    def __str__(self) -> str:
        return f"({self.e1} and {self.e2})"

@dataclass
class Or(Expr):
    e1: Expr
    e2: Expr

    def __str__(self) -> str:
        return f"({self.e1} or {self.e2})"

@dataclass
class Not(Expr):
    e1: Expr

    def __str__(self) -> str:
        return f"(not {self.e1})"

@dataclass
class Matrix(Expr):
    row: Annotated[int, IntRange(-1, 1)]
    column: Annotated[int, IntRange(-1, 1)]

    def __str__(self) -> str:
        return f"(X[{self.row}, {self.column}])"

def evaluate(e: Expr) -> Callable[[Any], float]:

    if isinstance(e, And):
        return lambda line: evaluate(e.e1)(line) and evaluate(e.e2)(line)
    elif isinstance(e, Or):
        return lambda line: evaluate(e.e1)(line) or evaluate(e.e2)(line)
    elif isinstance(e, Not):
        return lambda line: not evaluate(e.e1)(line)
    elif isinstance(e, Matrix):
        return lambda line: line[e.row, e.column]
    else:
        raise NotImplementedError(str(e))


def fitness_function(i: Expr):
    _clf = evaluate(i)
    ypred = [_clf(line) for line in np.rollaxis(Xtrain, 0)]
    return accuracy_score(ytrain, ypred)


def learn_predictor_gn(Xtrain, ytrain, Xtest, ytest):

    g = extract_grammar([And, Or, Not, Matrix], Expr)
    print(f"Grammar: {repr(g)}")

    alg = GP(
        g,
        treebased_representation,
        evaluation_function=fitness_function, 
        number_of_generations=5)

    (b, bf, bp) = alg.evolve(verbose=0)

    print("Best individual:", bp)
    print("Genetic Engine Train Accuracy:", bf)
    
    _clf = evaluate(bp)
    ypred = [_clf(line) for line in np.rollaxis(Xtest, 0)]
    print("GeneticEngine Test Accuracy:", accuracy_score(ytest, ypred))


if __name__ == "__main__":

    (Xtrain, ytrain) = generate_dataset(1000)
    (Xtest, ytest) = generate_dataset(1000)
    
    # # Generate dataset for PonyGE2
    # # Train
    # _x = Xtrain.reshape(1000, 9)
    # _y = ytrain.reshape(1000, 1)
    # np.savetxt("Train.csv", np.concatenate([_x, _y], axis=1), fmt='%i', delimiter=",")

    # # Test
    # _x = Xtest.reshape(1000, 9)
    # _y = ytest.reshape(1000, 1)
    # np.savetxt("Test.csv", np.concatenate([_x, _y], axis=1), fmt='%i', delimiter=",")
    
    #learn_predictor(Xtrain, ytrain, Xtest, ytest)
    learn_predictor_gn(Xtrain, ytrain, Xtest, ytest)

