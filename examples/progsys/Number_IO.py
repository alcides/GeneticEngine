from typing import Annotated, Any, Callable
import sys
from utils import get_data, import_embedded

from geneticengine.core.grammar import extract_grammar
from geneticengine.grammars.sgp import Plus, Literal, Number, Mul, SafeDiv, Var
from geneticengine.metahandlers.vars import VarRange
from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.representations.treebased import treebased_representation

FILE_NAME = "Number_IO"
DATA_FILE_TRAIN = "C:\\Users\\leoni\\Desktop\\Master\\Scriptie\\GeneticEngine\\examples\\progsys\\data\\{}\\Train.txt".format(FILE_NAME)
DATA_FILE_TEST = "C:\\Users\\leoni\\Desktop\\Master\\Scriptie\\GeneticEngine\\examples\\progsys\\data\\{}\\Test.txt".format(FILE_NAME)

inval,outval = get_data(DATA_FILE_TRAIN,DATA_FILE_TEST)
imported = import_embedded(FILE_NAME)

Var.__annotations__["name"] = Annotated[str, VarRange(["in0", "in1"])]
g = extract_grammar([Plus, Mul, SafeDiv, Literal, Var], Number)
print("Grammar: {}.".format(repr(g)))

def safediv(x, y):
    if y == 0:
        return 0.00001
    else:
        return x / y

variables = {}
for i, n in enumerate(["in0", "in1"]):
    variables[n] = i

def evaluate(n: Number) -> Callable[[Any], float]:
    if isinstance(n, Plus):
        return lambda line: evaluate(n.left)(line) + evaluate(n.right)(line)
    elif isinstance(n, Mul):
        return lambda line: evaluate(n.left)(line) * evaluate(n.right)(line)
    elif isinstance(n, SafeDiv):
        return lambda line: safediv(evaluate(n.left)(line), evaluate(n.right)(line))
    elif isinstance(n, Literal):
        return lambda _: n.val
    elif isinstance(n, Var):
        return lambda line: line[variables[n.name]]
    else:
        return lambda line: 0


def fitness_function(n):
    fitness, error, cases = imported.fitness(inval,outval,evaluate(n))
    return fitness


alg = GP(
    g,
    treebased_representation,
    fitness_function,
    minimize=True,
)
(b, bf, bp) = alg.evolve(verbose=0)
print(bf, bp, b)

