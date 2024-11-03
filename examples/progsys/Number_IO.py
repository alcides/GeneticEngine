from __future__ import annotations

from typing import Annotated

from examples.progsys.utils import get_data
from examples.progsys.utils import import_embedded

from geml.simplegp import SimpleGP
from geneticengine.grammar.grammar import extract_grammar
from geml.grammars.coding.numbers import Literal
from geml.grammars.coding.numbers import Mul
from geml.grammars.coding.numbers import Number
from geml.grammars.coding.numbers import Plus
from geml.grammars.coding.numbers import SafeDiv
from geml.grammars.coding.numbers import Var
from geneticengine.grammar.metahandlers.vars import VarRange

# ===================================
# This is a simple example on how to use GeneticEngine to solve a GP problem.
# We define the tree structure of the representation and then we define the fitness function for our problem
# This Number_IO example is a combinatorial optimization problem, given an integer and a float, calculate their sum.
# We used the Number_IO dataset stored in examples/progsys/data folder
# Problem taken from the following paper: https://dl.acm.org/doi/abs/10.1145/2739480.2754769
# ===================================

FILE_NAME = "Number_IO"
DATA_FILE_TRAIN = f"./examples/progsys/data/{FILE_NAME}/Train.txt"
DATA_FILE_TEST = f"./examples/progsys/data/{FILE_NAME}/Test.txt"

inval, outval = get_data(DATA_FILE_TRAIN, DATA_FILE_TEST)
imported = import_embedded(FILE_NAME)

vars = ["in0", "in1"]
variables = {}
for i, n in enumerate(vars):
    variables[n] = i

Var.__init__.__annotations__["name"] = Annotated[str, VarRange(vars)]
Var.feature_indices = variables  # type: ignore


def fitness_function(n: Number):
    fitness, error, cases = imported.fitness(inval, outval, n.evaluate_lines())
    return fitness


def preprocess():
    return extract_grammar([Plus, Mul, SafeDiv, Literal, Var], Number)


def evolve(g, seed, mode, representation="treebased"):
    alg = SimpleGP(
        grammar=g,
        representation=representation,
        minimize=True,
        fitness_function=fitness_function,
        max_evaluations=10000,
        seed=seed,
        max_depth=10,
        population_size=50,
        crossover_probability=0.9,
    )
    ind = alg.search()[0]
    return ind.get_phenotype(), ind.get_fitness(alg.get_problem()), g


if __name__ == "__main__":
    g = preprocess()
    ind = evolve(g, 0, False)
    print(ind)
