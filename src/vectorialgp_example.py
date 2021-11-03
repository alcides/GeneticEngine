from dataclasses import dataclass
from typing import Annotated, Protocol

import numpy as np
from sklearn.metrics import mean_squared_error

from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.treebased import treebased_representation
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.floats import FloatRange

# Load Dataset
dataset = [
    [1, 5, [1, 2, 3], [1, 5, 6], 0],
    [2, 2, [3, 2, 3], [1, 8, 6], 1],
    [1, 6, [2, 2, 3], [1, 5, 9], 2],
    [2, 8, [3, 2, 3], [1, 6, 6], 3],
]


class Scalar:
    pass


class Vectorial:
    pass


@dataclass
class Value(Scalar):
    value: Annotated[float, FloatRange(-10, 10)]


@dataclass
class ScalarVar(Scalar):
    index: Annotated[int, IntRange(0, 1)]


@dataclass
class VectorialVar(Vectorial):
    index: Annotated[int, IntRange(2, 3)]


@dataclass
class Add(Scalar):
    right: Scalar
    left: Scalar


@dataclass
class Mean(Scalar):
    arr: Vectorial


@dataclass
class CumulativeSum(Vectorial):
    arr: Vectorial


g = extract_grammar([Value, ScalarVar, VectorialVar, Add, Mean, CumulativeSum], Scalar)
print("Grammar: {}.".format(repr(g)))


def compile(p):
    if isinstance(p, Value):
        return lambda line: p.value
    elif isinstance(p, ScalarVar) or isinstance(p, VectorialVar):
        return lambda line: line[p.index]
    elif isinstance(p, Add):
        return lambda line: compile(p.left)(line) + compile(p.right)(line)
    elif isinstance(p, Mean):
        return lambda line: np.mean(compile(p.arr)(line))
    elif isinstance(p, CumulativeSum):
        return lambda line: np.cumsum(compile(p.arr)(line))
    else:
        raise NotImplementedError(str(p))


def fitness_function(n: Scalar):
    regressor = compile(n)
    y_pred = [regressor(line) for line in dataset]
    y = [line[-1] for line in dataset]
    r = mean_squared_error(y, y_pred)
    return r


def translate(p):
    if isinstance(p, Value):
        return str(p.value)
    elif isinstance(p, ScalarVar) or isinstance(p, VectorialVar):
        return "line[{}]".format(p.index)
    elif isinstance(p, Add):
        return "({} + {})".format(translate(p.left), translate(p.right))
    elif isinstance(p, Mean):
        return "np.mean({})".format(translate(p.arr))
    elif isinstance(p, CumulativeSum):
        return "np.cumsum({})".format(translate(p.arr))
    else:
        raise NotImplementedError(str(p))


def fitness_function_alternative(n: Scalar):
    code = "lambda line: {}".format(translate(n))
    regressor = eval(code)
    y_pred = [regressor(line) for line in dataset]
    y = [line[-1] for line in dataset]
    r = mean_squared_error(y, y_pred)
    return r


alg = GP(
    g,
    treebased_representation,
    fitness_function_alternative,
    minimize=True,
    seed=0,
    population_size=100,
    number_of_generations=100,
)
(b, bf) = alg.evolve()
print(bf, b)
