import os
import ast
from typing import Annotated, Any, Callable
import sys
sys.path.insert(1, 'C:\\Users\\leoni\\Desktop\\Master\\Scriptie\\GeneticEngine')

from geneticengine.core.grammar import extract_grammar
from geneticengine.grammars.sgp import Plus, Literal, Number, Mul, SafeDiv, Var
from geneticengine.metahandlers.vars import VarRange
from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.representations.treebased import treebased_representation



CWD = os.path.dirname(os.path.realpath(__file__))
FILE_NAME = "Number IO"
DATA_FILE_TRAIN = "C:\\Users\\leoni\\Desktop\\Master\\Scriptie\\GeneticEngine\\examples\\progsys\\data\\{}\\Train.txt".format(FILE_NAME)
DATA_FILE_TEST = "C:\\Users\\leoni\\Desktop\\Master\\Scriptie\\GeneticEngine\\examples\\progsys\\data\\{}\\Test.txt".format(FILE_NAME)

def bla():
    return "print(1 + 1)\nprint(2.5 * 2)"

def get_data(data_file_train,data_file_test):
    with open(data_file_train, 'r') as train_file, \
            open(data_file_test, 'r') as test_file:
        train_data = train_file.read()
        test_data = test_file.read()

    t = train_data.split('\n')

    inval = t[0].strip('inval = ')
    outval = t[1].strip('outval = ')
    inval = ast.literal_eval(inval)
    outval = ast.literal_eval(outval)
    return inval,outval


inval,outval = get_data(DATA_FILE_TRAIN,DATA_FILE_TEST)
imported = __import__(FILE_NAME + "-Embed")

evolved_function = lambda x,y: x+y


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



