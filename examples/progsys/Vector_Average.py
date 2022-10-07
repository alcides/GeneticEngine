from __future__ import annotations

from typing import Annotated
from typing import Any
from typing import Callable

import geneticengine.grammars.coding.lists as lists
import geneticengine.grammars.coding.numbers as numbers
from examples.progsys.utils import get_data
from examples.progsys.utils import import_embedded
from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.grammatical_evolution.ge import (
    ge_representation,
)
from geneticengine.core.representations.grammatical_evolution.structured_ge import (
    sge_representation,
)
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.grammars.coding.classes import Number
from geneticengine.grammars.coding.classes import Statement
from geneticengine.grammars.coding.classes import XAssign
from geneticengine.grammars.coding.conditions import Equals
from geneticengine.grammars.coding.conditions import GreaterOrEqualThan
from geneticengine.grammars.coding.conditions import GreaterThan
from geneticengine.grammars.coding.conditions import Is
from geneticengine.grammars.coding.conditions import IsNot
from geneticengine.grammars.coding.conditions import LessOrEqualThan
from geneticengine.grammars.coding.conditions import LessThan
from geneticengine.grammars.coding.conditions import NotEquals
from geneticengine.grammars.coding.control_flow import IfThen
from geneticengine.grammars.coding.control_flow import IfThenElse
from geneticengine.grammars.coding.control_flow import While
from geneticengine.grammars.coding.logical_ops import And
from geneticengine.grammars.coding.logical_ops import Or
from geneticengine.metahandlers.vars import VarRange

# Max, Min, Abs, Plus, Literal, Mul, SafeDiv, Var
# Max, Min, Abs, Plus, Literal, Mul, SafeDiv, Var

FILE_NAME = "Vector_Average"
DATA_FILE_TRAIN = f"./examples/progsys/data/{FILE_NAME}/Train.txt"
DATA_FILE_TEST = f"./examples/progsys/data/{FILE_NAME}/Test.txt"

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


def evolve(g, seed, mode, representation=""):
    if representation == "ge":
        representation = ge_representation
    elif representation == "sge":
        representation = sge_representation
    else:
        representation = treebased_representation
    alg = GP(
        g,
        fitness_function,
        representation=representation,
        number_of_generations=5,
        minimize=True,
        seed=seed,
        max_depth=10,
        population_size=50,
        probability_crossover=0.9,
        timer_stop_criteria=mode,
    )
    (b, bf, bp) = alg.evolve(verbose=1)
    return b, bf


if __name__ == "__main__":
    g = preprocess()
    bf, b = evolve(g, 0, False)
    print(b)
    print(f"With fitness: {bf}")
