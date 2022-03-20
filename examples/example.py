from __future__ import annotations

from dataclasses import dataclass
from random import random
from typing import Annotated

from geneticengine.algorithms.gp.gp import GP
from geneticengine.algorithms.hill_climbing import HC
from geneticengine.algorithms.random_search import RandomSearch
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.grammars.sgp import Literal
from geneticengine.grammars.sgp import Mul
from geneticengine.grammars.sgp import Number
from geneticengine.grammars.sgp import Plus
from geneticengine.grammars.sgp import SafeDiv
from geneticengine.grammars.sgp import Var
from geneticengine.metahandlers.vars import VarRange

Var.__init__.__annotations__["name"] = Annotated[str, VarRange("x")]
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

alg_gp = GP(
    g,
    fit,
    representation=treebased_representation,
    population_size=150,
    max_depth=5,
    number_of_generations=10,
    minimize=True,
    n_elites=10,
    n_novelties=10,
)
(b_gp, bf_gp, bp_gp) = alg_gp.evolve(verbose=1)


alg_hc = HC(
    g,
    evaluation_function=fit,
    representation=treebased_representation,
    population_size=150,
    max_depth=5,
    number_of_generations=10,
    minimize=True,
)
(b_hc, bf_hc, bp_hc) = alg_hc.evolve(verbose=1)

alg_rs = RandomSearch(
    g,
    evaluation_function=fit,
    representation=treebased_representation,
    population_size=150,
    max_depth=5,
    number_of_generations=40,
    minimize=True,
    favor_less_deep_trees=True,
)
(b_rs, bf_rs, bp_rs) = alg_rs.evolve(verbose=1)

print("\n======\nRS\n======\n")
print(bf_rs, bp_rs, b_rs)

print("\n======\nHC\n======\n")
print(bf_hc, bp_hc, b_hc)

print("\n======\nGP\n======\n")
print(bf_gp, bp_gp, b_gp)
