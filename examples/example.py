from __future__ import annotations

from random import random
from typing import Annotated

from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.algorithms.hill_climbing import HC
from geneticengine.algorithms.random_search import RandomSearch
from geneticengine.evaluation.budget import TimeBudget
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geml.grammars.basic_math import SafeDiv
from geml.grammars.sgp import Literal
from geml.grammars.sgp import Mul
from geml.grammars.sgp import Number
from geml.grammars.sgp import Plus
from geml.grammars.sgp import Var
from geneticengine.grammar.metahandlers.vars import VarRange

# ===================================
# This is a simple example on how to use GeneticEngine to solve a GP problem.
# We define the tree structure of the representation and then we define the fitness function for our problem
# In this example we are solving the same problem using three different algoritms; Hill Climbing , Random Search and a GP algorithm
# ===================================

Var.__init__.__annotations__["name"] = Annotated[str, VarRange(["x"])]
g = extract_grammar([Plus, Mul, SafeDiv, Literal, Var], Number)
print("Grammar:")
print(repr(g))


def fit(p):
    error = 0
    for _ in range(100):
        n = random() * 100
        goal = target(n)
        got = p.evaluate(x=n)
        error += (goal - got) ** 2
    return error


def target(x):
    return x**2


r = NativeRandomSource()
representation = TreeBasedRepresentation(g, decider=MaxDepthDecider(r, g, 5))
problem = SingleObjectiveProblem(
    minimize=True,
    fitness_function=fit,
)
budget = TimeBudget(3)


alg_gp = GeneticProgramming(problem=problem, budget=budget, representation=representation)
ind = alg_gp.search()[0]
print("\n======\nGP\n======\n")
print(f"{ind.get_fitness(problem)} - {ind}")


alg_hc = HC(problem=problem, budget=budget, representation=representation)
ind = alg_hc.search()[0]
print("\n======\nHC\n======\n")
print(f"{ind.get_fitness(problem)} - {ind}")

alg_rs = RandomSearch(problem=problem, budget=budget, representation=representation)
ind = alg_rs.search()[0]
print("\n======\nRS\n======\n")
print(f"{ind.get_fitness(problem)} - {ind}")
