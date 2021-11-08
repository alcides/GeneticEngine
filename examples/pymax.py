from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from textwrap import indent
from typing import Annotated, List, Protocol, Union
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.treebased import treebased_representation
from geneticengine.core.tree import TreeNode
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.lists import ListSizeBetween
from geneticengine.algorithms.gp.gp import GP
from number_maker import Mul


class Statment:
    pass


@dataclass
class Code:
    stmts: List[Statment]


class Expr:
    def evaluate(self, **kwargs) -> float:
        return 0.0


@dataclass
class ForLoop(Expr):
    iterationRange: Annotated[int, IntRange(1, 6)]
    loopedCode: Expr

    def evaluate(self, **kwargs):
        x = self.loopedCode
        if x.__class__ == OneHalve:
            y = deepcopy(x)
        elif x.__class__ == ForLoop:
            y = ForLoop(self.iterationRange * x.iterationRange, x.loopedCode)
        else:
            y = deepcopy(x)
            if x.__class__ == PlusOneHalve:
                for _ in range(self.iterationRange):
                    # Add recursiveness
                    y = Plus(deepcopy(x), deepcopy(y))
        return y.evaluate()

    def __str__(self):
        return "for i in range({}):\n{}".format(
            self.iterationRange, indent(str(self.loopedCode), "\t")
        )


class Const(Expr):
    def evaluate(self, **kwargs):
        return 0.5

    def __str__(self) -> str:
        return "0.5"


@dataclass
class PlusOneHalve(Expr):
    left: Expr
    right: Const

    def evaluate(self, **kwargs):
        return self.left.evaluate(**kwargs) + self.right.evaluate(**kwargs)

    def __str__(self) -> str:
        return "{} + {}".format(self.left, self.right)


@dataclass
class MultOneHalve(Expr):
    left: Expr

    def evaluate(self, **kwargs):
        return self.left.evaluate() * OneHalve().evaluate()

    def __str__(self) -> str:
        return "x = (" + "x" + " * " + str(OneHalve()) + ")"


def fit(indiv):
    code = "x = 0\n" + str(indiv)
    loc = {}
    exec(code, globals(), loc)
    x = loc["x"]
    return x


fitness_function = lambda x: fit(x)

if __name__ == "__main__":
    g = extract_grammar([PlusOneHalve, MultOneHalve, ForLoop], Expr)
    alg = GP(
        g,
        treebased_representation,
        fitness_function,
        max_depth=5,
        max_init_depth=5,
        population_size=40,
        number_of_generations=3,
        minimize=False,
    )
    (b, bf) = alg.evolve(verbose=0)
    print(b)
    print("With fitness: {}".format(bf))
