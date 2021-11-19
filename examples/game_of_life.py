from abc import ABC
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Tuple
import numpy as np
from numpy.lib.arraysetops import isin

from sklearn.tree import DecisionTreeClassifier

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

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(Xtrain.reshape(1000, 9), ytrain)
    ypred = clf.predict(Xtest.reshape(1000, 9))
    accuracy = (ypred == ytest).mean()

    print("Decision Tree Test Accuracy:", accuracy)


def learn_predictor_gn(Xtrain, ytrain, Xtest, ytest):

    # TODO: Pedro, implementar um classificador com:

    """
    We are building a rule that returns a binary expression.
    A binary expression can have one of the following shapes:

    b1 and b2 (where bn is another binary expression)
    b1 or b2
    not b1
    matrix[i, j] (where i and j are integers between -1 and 1, inclusive)

    Each binary expression should be evaluated for each instance of the dataset.

    You can use the Xtrain and ytrain to evolve a GeneticProgramming algorithm,
    and you will use the best individual of a population of 50 after 100 generations
    to predict the test set.

    """

    train_accuracy = 0
    print("Genetic Engine Train Accuracy: {}", train_accuracy)
    accuracy = 0
    print("GeneticEngine Test Accuracy:", accuracy)


if __name__ == "__main__":
    (Xtrain, ytrain) = generate_dataset(1000)
    (Xtest, ytest) = generate_dataset(1000)
    learn_predictor(Xtrain, ytrain, Xtest, ytest)
    learn_predictor_gn(Xtrain, ytrain, Xtest, ytest)
