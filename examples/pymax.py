from __future__ import annotations

import sys
from dataclasses import dataclass

from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.grammatical_evolution import ge_representation
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.grammars.coding.classes import Expr
from geneticengine.grammars.coding.classes import Statement
from geneticengine.grammars.coding.classes import XAssign
from geneticengine.grammars.coding.control_flow import Code
from geneticengine.grammars.coding.control_flow import ForLoop


class VarX(Expr):
    def evaluate(self, x=0):
        return x

    def __str__(self) -> str:
        return "x"


class Const(Expr):
    def evaluate(self, x=0):
        return 0.5

    def __str__(self) -> str:
        return "0.5"


@dataclass
class XPlusConst(Expr):
    right: Const

    def evaluate(self, x):
        return x + self.right.evaluate(x)

    def __str__(self) -> str:
        return f"x + {self.right}"


@dataclass
class XTimesConst(Expr):
    right: Const

    def evaluate(self, x):
        return x * self.right.evaluate(x)

    def __str__(self) -> str:
        return f"x * {self.right}"


def fit(indiv: Code):
    return indiv.evaluate()


def fitness_function(x):
    return fit(x)


def preprocess():
    return extract_grammar(
        [XPlusConst, XTimesConst, XAssign, ForLoop, Code, Const, VarX],
        ForLoop,
    )


def evolve(
    g,
    seed,
    mode,
    representation="treebased_representation",
):
    if representation == "grammatical_evolution":
        representation = ge_representation
    else:
        representation = treebased_representation

    alg = GP(
        g,
        fitness_function,
        representation=representation,
        max_depth=13,
        population_size=25,
        number_of_generations=10,
        minimize=False,
        timer_stop_criteria=mode,
    )
    (b, bf, bp) = alg.evolve(verbose=1)
    return b, bf


if __name__ == "__main__":
    g = preprocess()
    bf, b = evolve(g, 0, False)
    print(b)
    print(f"Final fitness: {bf}")
