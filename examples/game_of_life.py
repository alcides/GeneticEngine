from __future__ import annotations

import csv
import os
import time
from abc import ABC
from dataclasses import dataclass
from optparse import OptionParser
from typing import Annotated
from typing import Any
from typing import Callable
from typing import Tuple

import global_vars as gv
import numpy as np
from sklearn.metrics import f1_score

from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.grammar import extract_grammar
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
from geneticengine.grammars.coding.logical_ops import And
from geneticengine.grammars.coding.logical_ops import Not
from geneticengine.grammars.coding.logical_ops import Or
from geneticengine.metahandlers.ints import IntRange

DATASET_NAME = "GameOfLife"
DATA_FILE_TRAIN = f"GeneticEngine/examples/data/{DATASET_NAME}/Train.csv"
DATA_FILE_TEST = f"GeneticEngine/examples/data/{DATASET_NAME}/Test.csv"
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


def preprocess():
    grammar = extract_grammar([And, Or, Not, MatrixElement], Condition)
    print(grammar)
    return grammar


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

    g = preprocess()
    alg = GP(
        g,
        fitness_function,
        representation=representation,
        probability_crossover=gv.PROBABILITY_CROSSOVER,
        probability_mutation=gv.PROBABILITY_MUTATION,
        number_of_generations=gv.NUMBER_OF_GENERATIONS,
        max_depth=gv.MAX_DEPTH,
        population_size=gv.POPULATION_SIZE,
        selection_method=gv.SELECTION_METHOD,
        n_elites=gv.N_ELITES,
        # ----------------
        minimize=False,
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
