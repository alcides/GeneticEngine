from dataclasses import dataclass
from random import random

from geneticengine.algorithms.gp import GP, create_tournament
from geneticengine.grammars.sgp import Plus, Literal, Number, Mul, SafeDiv, Var
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.tree import Node
from geneticengine.core.representations.treebased import treebased_representation


@dataclass
class Zero(Node, Number):
    def evaluate(self, **kwargs):
        return 0


g = extract_grammar([Plus, Mul, SafeDiv, Literal, Var, Zero], Number)
print("Grammar:")
print(repr(g))

def fit(p):
    error = 0
    for _ in range(100):
        n = random() * 100
        m = random() * 100
        goal = target(n,m,0)
        got = p.evaluate(x=n, y=0,z=0)
        error += (goal -got)**2
    return error

def target(x,y,z):
    return x**2 

# target = 234.5
# fitness_function = lambda p: (abs(target - p.evaluate(x=1, y=2, z=3)))

alg = GP(
    g,
    treebased_representation,
    fit,
    number_of_generations=100,
    minimize=True,
)
(b, bf) = alg.evolve()
print(bf, b)
