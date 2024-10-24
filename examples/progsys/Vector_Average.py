from __future__ import annotations

from typing import Annotated

from examples.progsys.utils import get_data
from examples.progsys.utils import import_embedded

import geml.grammars.coding.lists as lists
import geml.grammars.coding.numbers as numbers
from geml.simplegp import SimpleGP
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.problems import SingleObjectiveProblem
from geml.grammars.coding.classes import Number
from geml.grammars.coding.classes import Statement
from geml.grammars.coding.classes import XAssign
from geml.grammars.coding.conditions import Equals
from geml.grammars.coding.conditions import GreaterOrEqualThan
from geml.grammars.coding.conditions import GreaterThan
from geml.grammars.coding.conditions import Is
from geml.grammars.coding.conditions import IsNot
from geml.grammars.coding.conditions import LessOrEqualThan
from geml.grammars.coding.conditions import LessThan
from geml.grammars.coding.conditions import NotEquals
from geml.grammars.coding.control_flow import IfThen
from geml.grammars.coding.control_flow import IfThenElse
from geml.grammars.coding.logical_ops import And
from geml.grammars.coding.logical_ops import Or
from geneticengine.grammar.metahandlers.vars import VarRange


# ===================================
# This is a simple example on how to use GeneticEngine to solve a GP problem.
# We define the tree structure of the representation and then we define the fitness function for our problem
# This Vector_Average example is a combinatorial optimization problem, given a vector of floats, returns the average of those floats.
# We used the Vector_Average dataset stored in examples/progsys/data folder
# Problem taken from the following paper: https://dl.acm.org/doi/abs/10.1145/2739480.2754769
# ===================================


# Max, Min, Abs, Plus, Literal, Mul, SafeDiv, Var
# Max, Min, Abs, Plus, Literal, Mul, SafeDiv, Var

FILE_NAME = "Vector_Average"
DATA_FILE_TRAIN = f"./data/{FILE_NAME}/Train.txt"
DATA_FILE_TEST = f"./data/{FILE_NAME}/Test.txt"

inval, outval = get_data(DATA_FILE_TRAIN, DATA_FILE_TEST)
imported = import_embedded(FILE_NAME)

vars = ["in0"]
variables = {}
for i, n in enumerate(vars):
    variables[n] = i

XAssign.__init__.__annotations__["value"] = Number
lists.Var.__init__.__annotations__["name"] = Annotated[str, VarRange(vars)]
lists.Var.feature_indices = variables  # type: ignore


def preprocess():
    return extract_grammar(
        [
            numbers.Plus,
            numbers.Literal,
            numbers.Mul,
            numbers.SafeDiv,
            numbers.Max,
            numbers.Min,
            numbers.Abs,
            lists.Length,
            lists.Literal,
            lists.Combine,
            lists.GetElement,
            lists.Max,
            lists.Min,
            And,
            Or,
            lists.Var,
            Equals,
            NotEquals,
            GreaterOrEqualThan,
            GreaterThan,
            LessOrEqualThan,
            LessThan,
            Is,
            IsNot,
            XAssign,
            IfThen,
            IfThenElse,  # , While
        ],
        Statement,
    )


def fitness_function(n: Statement):
    fitness, error, cases = imported.fitness(inval, outval, n.evaluate_lines())
    return fitness


def evolve(g, seed, mode, representation="treebased"):
    alg = SimpleGP(
        grammar=g,
        representation=representation,
        problem=SingleObjectiveProblem(
            minimize=True,
            fitness_function=fitness_function,
        ),
        max_evaluations=10000,
        seed=seed,
        max_depth=10,
        population_size=50,
        crossover_probability=0.9,
    )
    ind = alg.search()[0]
    return ind.get_phenotype(), ind.fitness, g


if __name__ == "__main__":
    g = preprocess()
    ind = evolve(g, 0, False)
    print(ind)
