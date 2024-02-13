from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated
from typing import Any
from typing import Callable

import numpy as np
from sklearn.metrics import f1_score

from geml.simplegp import SimpleGP
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.grammar import Grammar
from geneticengine.problems import Problem
from geneticengine.problems import SingleObjectiveProblem
from geml.grammars.coding.classes import Condition
from geml.grammars.coding.classes import Expr
from geml.grammars.coding.logical_ops import And
from geml.grammars.coding.logical_ops import Not
from geml.grammars.coding.logical_ops import Or
from geneticengine.grammar.metahandlers.ints import IntRange

# ===================================
# This is an example on how to use GeneticEngine to solve a GP problem.
# We define the tree structure of the representation and then we define the fitness function for our problem
# In this example we are Reversing Game of Life using a Vectorial Approach to Genetic Programming
# We used the GameOfLife dataset stored in examples/data folder
# ===================================

DATASET_NAME = "GameOfLife"
DATA_FILE_TRAIN = f"examples/data/{DATASET_NAME}/Train.csv"
DATA_FILE_TEST = f"examples/data/{DATASET_NAME}/Test.csv"
OUTPUT_FOLDER = "GoL/grammar_standard"


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


class GameOfLifeBenchmark:
    def get_problem(self) -> Problem:
        return SingleObjectiveProblem(
            minimize=False,
            fitness_function=fitness_function,
        )

    def get_grammar(self) -> Grammar:
        return extract_grammar([And, Or, Not, MatrixElement], Condition)

    def main(self, **args):
        g = self.get_grammar()
        prob = self.get_problem()
        alg = SimpleGP(
            g,
            problem=prob,
            number_of_generations=50,
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
            f"Fitness of {best.get_fitness(prob)} by genotype: {best.genotype} with phenotype: {best.get_phenotype()}",
        )

        _clf = evaluate(best.get_phenotype())
        ypred = [_clf(line) for line in np.rollaxis(Xtest, 0)]
        print("GeneticEngine Test F1 score:", f1_score(ytest, ypred))


if __name__ == "__main__":
    GameOfLifeBenchmark().main(seed=0)
