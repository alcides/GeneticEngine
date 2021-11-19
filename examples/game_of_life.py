from abc import ABC
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Tuple
import numpy as np
from numpy.lib.arraysetops import isin

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


def learn_predictor():
    (Xtrain, ytrain) = generate_dataset(1000)
    (Xtest, ytest) = generate_dataset(1000)

    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(Xtrain.reshape(1000, 9), ytrain)
    ypred = clf.predict(Xtest.reshape(1000, 9))
    accuracy = (ypred == ytest).mean()

    print("Decision Tree Test Accuracy:", accuracy)


class B(ABC):
    pass


@dataclass
class And(B):
    l: B
    r: B


@dataclass
class Or(B):
    l: B
    r: B


@dataclass
class Not(B):
    v: B


@dataclass
class Matrix(B):
    i: Annotated[int, IntRange(-1, 2)]
    j: Annotated[int, IntRange(-1, 2)]


def compile(b: B) -> Callable[[Any], int]:
    if isinstance(b, And):
        return lambda m: compile(b.l)(m) and compile(b.r)(m)
    elif isinstance(b, Or):
        return lambda m: compile(b.l)(m) or compile(b.r)(m)
    elif isinstance(b, Not):
        return lambda m: 1 - compile(b.v)(m)
    else:
        assert isinstance(b, Matrix)
        return lambda x: x[b.i, b.j]


def learn_predictor_gn():
    (Xtrain, ytrain) = generate_dataset(1000)
    (Xtest, ytest) = generate_dataset(1000)

    def accuracy_f(x, y):
        return (x == y).mean()

    def fitness_function(b: B):
        fake_game_of_life_rule = compile(b)
        ypred = np.fromiter((fake_game_of_life_rule(xi) for xi in Xtrain), Xtrain.dtype)
        return accuracy_f(ypred, ytrain)

    g = extract_grammar([And, Or, Not, Matrix], B)
    alg = GP(
        g,
        treebased_representation,
        fitness_function,
        max_depth=10,
        population_size=50,
        number_of_generations=100,
        minimize=False,
    )
    (b, bf, bp) = alg.evolve(verbose=False)
    print(bp, b)
    print("Genetic Engine Train Accuracy: {}".format(bf))

    assert isinstance(bp, B)
    fake_game_of_life_rule = compile(bp)
    ypred = np.fromiter((fake_game_of_life_rule(xi) for xi in Xtrain), Xtrain.dtype)
    accuracy = (ypred == ytest).mean()
    print("GeneticEngine Test Accuracy:", accuracy)


if __name__ == "__main__":
    learn_predictor()
    learn_predictor_gn()
