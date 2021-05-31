from geneticengine.core.random.sources import RandomSource
from geneticengine.grammars.sgp import Plus, Literal, Number
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.treebased import random_individual

g = extract_grammar([Plus, Literal], Number)
print(repr(g))

i = random_individual(RandomSource(), g, 5)

print(i)