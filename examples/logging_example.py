from __future__ import annotations

from random import random
from typing import Annotated
import logging

from geneticengine.algorithms.gp.gp import GeneticProgramming
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

# Setup logging

logger = logging.getLogger("geneticengine")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


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
representation = TreeBasedRepresentation(g, MaxDepthDecider(r, g, 5))
problem = SingleObjectiveProblem(
    minimize=True,
    fitness_function=fit,
)
budget = TimeBudget(3)


alg_gp = GeneticProgramming(problem=problem, budget=budget, representation=representation, population_size=10)
ind = alg_gp.search()[0]
print("\n======\nGP\n======\n")
print(f"{ind.get_fitness(problem)} - {ind}")
