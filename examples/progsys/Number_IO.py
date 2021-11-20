from typing import Annotated, Any, Callable
import sys
from utils import get_data, import_embedded

from geneticengine.core.grammar import extract_grammar
from geneticengine.grammars.sgp import Plus, Literal, Number, Mul, SafeDiv, Var
from geneticengine.metahandlers.vars import VarRange
from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.representations.treebased import treebased_representation

FILE_NAME = "Number_IO"
DATA_FILE_TRAIN = "./examples/progsys/data/{}/Train.txt".format(FILE_NAME)
DATA_FILE_TEST = "./examples/progsys/data/{}/Test.txt".format(FILE_NAME)

inval,outval = get_data(DATA_FILE_TRAIN,DATA_FILE_TEST)
imported = import_embedded(FILE_NAME)

variables = {}
for i, n in enumerate(["in0", "in1"]):
    variables[n] = i

Var.__annotations__["name"] = Annotated[str, VarRange(["in0", "in1"])]
Var.feature_indices = variables
g = extract_grammar([Plus, Mul, SafeDiv, Literal, Var], Number)
print("Grammar: {}.".format(repr(g)))


def fitness_function(n: Number):
    fitness, error, cases = imported.fitness(inval,outval,n.evaluate_lines())
    return fitness


alg = GP(
    g,
    treebased_representation,
    fitness_function,
    minimize=True,
)
(b, bf, bp) = alg.evolve(verbose=0)
print(bf, bp, b)

