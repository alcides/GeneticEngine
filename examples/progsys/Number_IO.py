from __future__ import annotations

from typing import Annotated

from utils import get_data
from utils import import_embedded

from geneticengine.algorithms.gp.simplegp import SimpleGP
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.representations.grammatical_evolution.ge import (
    GrammaticalEvolutionRepresentation,
)
from geneticengine.core.representations.grammatical_evolution.structured_ge import (
    StructuredGrammaticalEvolutionRepresentation,
)
from geneticengine.core.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.grammars.coding.numbers import Literal
from geneticengine.grammars.coding.numbers import Mul
from geneticengine.grammars.coding.numbers import Number
from geneticengine.grammars.coding.numbers import Plus
from geneticengine.grammars.coding.numbers import SafeDiv
from geneticengine.grammars.coding.numbers import Var
from geneticengine.metahandlers.vars import VarRange

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


def evolve(g, seed, mode, representation=""):
    if representation == "ge":
        representation = GrammaticalEvolutionRepresentation
    elif representation == "sge":
        representation = GrammaticalEvolutionRepresentation
    else:
        representation = TreeBasedRepresentation
    alg = SimpleGP(
        g,
        representation=representation,
        problem=SingleObjectiveProblem(
            minimize=True,
            fitness_function=fitness_function,
            target_fitness=None,
        ),
        number_of_generations=50,
        seed=seed,
        max_depth=10,
        population_size=50,
        probability_crossover=0.9,
        timer_stop_criteria=mode,
    )
    (b, bf, bp) = alg.evolve()
    return b, bf


if __name__ == "__main__":
    g = preprocess()
    bf, b = evolve(g, 0, False)
    print(b)
    print(f"With fitness: {bf}")
