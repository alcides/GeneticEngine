from __future__ import annotations

from typing import Annotated

import numpy as np

from geneticengine.algorithms.gp.simplegp import SimpleGP
from geneticengine.core.grammar import extract_grammar
from geneticengine.grammars.sgp import Literal
from geneticengine.grammars.sgp import Mul
from geneticengine.grammars.sgp import Number
from geneticengine.grammars.sgp import Plus
from geneticengine.grammars.sgp import simplify
from geneticengine.grammars.sgp import Var
from geneticengine.metahandlers.vars import VarRange

SEED = 100


equation = lambda x: x * x + 1
array_size = 1000

np.random.seed(SEED)

X_train = np.random.randint(-100, 100, size=array_size)
noise = np.random.random(size=array_size)
y_train = np.vectorize(equation)(X_train) + noise

X_test = np.random.randint(-100, 100, size=array_size)
y_test = np.vectorize(equation)(X_test)


def fitness(p) -> float:
    y_pred = p.evaluate(X=X_train)
    diff = np.abs(y_pred - y_train)
    return np.mean(diff)


def fitness_test(p) -> float:
    y_pred = p.evaluate(X=X_test)
    diff = np.abs(y_pred - y_test)
    return np.mean(diff)


Var.__init__.__annotations__["name"] = Annotated[str, VarRange(["X"])]

g = extract_grammar([Mul, Plus, Var, Literal], Number)

gp = SimpleGP(
    grammar=g,
    evaluation_function=fitness,
    minimize=True,
    number_of_generations=10,
    save_to_csv="scratch.csv",
    seed=SEED,
    test_data=fitness_test,
)

p, _, _ = gp.evolve()
print(simplify(p.get_phenotype()))
