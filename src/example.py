from geneticengine.algorithms.gp import GP, create_tournament
from geneticengine.grammars.sgp import Plus, Literal, Number, Mul, SafeDiv, Var
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.treebased import treebased_representation

g = extract_grammar([Plus, Mul, SafeDiv, Literal, Var], Number)
print("Grammar:")
print(repr(g))


fitness_function = lambda x: x.evaluate(x=1, y=2, z=3)

alg = GP(
    g,
    treebased_representation,
    fitness_function,
)
(b, bf) = alg.evolve()
print(bf, b)
