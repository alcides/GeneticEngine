from dataclasses import dataclass
from random import random
from typing import Annotated

from geneticengine.algorithms.gp.gp import GP
from geneticengine.grammars.sgp import Plus, Literal, Number, Mul, SafeDiv, Var
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.treebased import treebased_representation
from geneticengine.metahandlers.vars import VarRange



Var.__annotations__["name"] = Annotated[str, VarRange("x")]
g = extract_grammar([Plus, Mul, SafeDiv, Literal, Var], Number)
print("Grammar:")
print(repr(g))


def fit(p):
    error = 0
    for _ in range(100):
        n = random() * 100
        m = random() * 100
        goal = target(n)
        got = p.evaluate(x=n)
        error += (goal - got) ** 2
    return error


def target(x):
    return x ** 2


# target = 234.5
# fitness_function = lambda p: (abs(target - p.evaluate(x=1, y=2, z=3)))

alg = GP(
    g,
    treebased_representation,
    fit,
    population_size=150,
    max_depth=5,
    number_of_generations=10,
    minimize=True,
    n_elites=10,
    n_novelties=10,
)
(b, bf, bp) = alg.evolve(verbose=0)
print(bf, bp, b)
