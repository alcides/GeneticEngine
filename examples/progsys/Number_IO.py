from __future__ import annotations

from typing import Annotated

from utils import get_data
from utils import import_embedded

from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.representations.grammatical_evolution.ge import (
    ge_representation,
)
from geneticengine.core.representations.grammatical_evolution.structured_ge import (
    sge_representation,
)
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.grammars.coding.numbers import Literal
from geneticengine.grammars.coding.numbers import Mul
from geneticengine.grammars.coding.numbers import Number
from geneticengine.grammars.coding.numbers import Plus
from geneticengine.grammars.coding.numbers import SafeDiv
from geneticengine.grammars.coding.numbers import Var
from geneticengine.metahandlers.vars import VarRange

FILE_NAME = "Number_IO"
DATA_FILE_TRAIN = f"./data/{FILE_NAME}/Train.txt"
DATA_FILE_TEST = f"./data/{FILE_NAME}/Test.txt"

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


def evolve(g, seed, mode, representation=""):
    if representation == "ge":
        representation = ge_representation
    elif representation == "sge":
        representation = sge_representation
    else:
        representation = treebased_representation
    alg = GP(
        g,
        representation=representation,
        problem=SingleObjectiveProblem(
            minimize=True,
            fitness_function=fitness_function,
            target_fitness=None,
        ),
        number_of_generations=50,
        seed=seed,
        max_depth=17,
        population_size=500,
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
