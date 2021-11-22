from abc import ABC
from dataclasses import dataclass
from textwrap import indent
from typing import Annotated, List, NamedTuple, Protocol
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.treebased import treebased_representation
from geneticengine.metahandlers.ints import IntRange
from geneticengine.algorithms.gp.gp import GP
from geneticengine.grammars.coding.control_flow import ForLoop, Code
from geneticengine.grammars.coding.expressions import XAssign
from geneticengine.grammars.coding.classes import Expr, Statement


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
        return "x + {}".format(self.right)


@dataclass
class XTimesConst(Expr):
    right: Const

    def evaluate(self, x):
        return x * self.right.evaluate(x)

    def __str__(self) -> str:
        return "x * {}".format(self.right)


def fit(indiv: Code):
    return indiv.evaluate()


fitness_function = lambda x: fit(x)


def preprocess():
    return extract_grammar(
        [XPlusConst, XTimesConst, XAssign, ForLoop, Code, Const, VarX], Code)


def evolve(g, seed):
    alg = GP(
        g,
        treebased_representation,
        fitness_function,
        max_depth=10,
        population_size=40,
        number_of_generations=15,
        minimize=False,
        seed=seed,
    )
    (b, bf, bp) = alg.evolve(verbose=0)
    return b, bf


if __name__ == "__main__":
    g = preprocess()
    bf, b = evolve(g, 0)
    print(b)
    print(f"With fitness: {bf}")
