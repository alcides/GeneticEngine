from geneticengine.algorithms.random_search import RandomSearch
from geneticengine.grammars.sgp import Plus, Literal, Number, Mul, SafeDiv, Var
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.treebased import random_individual

g = extract_grammar([Plus, Mul, SafeDiv, Literal, Var], Number)
print("Grammar:")
print(repr(g))


alg = RandomSearch(g, random_individual, lambda x: x.evaluate(x=1, y=2, z=3))
(b, bf) = alg.evolve()
print(bf, b)
