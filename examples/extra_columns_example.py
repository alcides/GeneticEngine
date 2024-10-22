from __future__ import annotations

from typing import Annotated

import numpy as np

from geml.simplegp import SimpleGP
from geneticengine.grammar.grammar import extract_grammar
from geml.grammars.sgp import Literal
from geml.grammars.sgp import Mul
from geml.grammars.sgp import Number
from geml.grammars.sgp import Plus
from geml.grammars.sgp import simplify
from geml.grammars.sgp import Var
from geneticengine.grammar.metahandlers.vars import VarRange

SEED = 100


def equation(x):
    return x * x + 1


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
    fitness_function=fitness,
    minimize=True,
    max_evaluations=10 * 100,
    csv_output="scratch.csv",
    csv_extra_fields={"test_data": lambda p: str(fitness_test(p))},
    seed=SEED,
)

ind = gp.search()[0]
print(simplify(ind.get_phenotype()))
