from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, List, Protocol, Union
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.treebased import treebased_representation
from geneticengine.core.tree import Node
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.lists import ListSizeBetween
from geneticengine.algorithms.gp.gp import GP


class Expr:
    pass


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


class OneHalve(Expr):
    def evaluate(self, **kwargs):
        return 0.5

    def __str__(self) -> str:
        return "0.5"


@dataclass
class Plus(Expr):
    left: Expr
    right: Expr

    def evaluate(self, **kwargs):
        return self.left.evaluate() + self.right.evaluate()

    def __str__(self) -> str:
        return "(" + str(self.left) + " + " + str(self.right) + ")"


@dataclass
class PlusOneHalve(Expr):
    left: Expr

    def evaluate(self, **kwargs):
        return self.left.evaluate() + OneHalve().evaluate()

    def __str__(self) -> str:
        return "(" + str(self.left) + " + " + str(OneHalve()) + ")"


@dataclass
class Mult(Expr):
    left: Expr

    def evaluate(self, **kwargs):
        return self.left.evaluate() * OneHalve().evaluate()

    def __str__(self) -> str:
        return "(" + str(self.left) + " * " + str(OneHalve()) + ")"


fitness_function = lambda x: x.evaluate()

if __name__ == "__main__":
    g = extract_grammar([OneHalve, PlusOneHalve, ForLoop], Expr)
    alg = GP(
        g,
        treebased_representation,
        fitness_function,
        max_depth=5,
        population_size=40,
        number_of_generations=3,
        minimize=False,
    )

    (b, bf) = alg.evolve()
    print(b, bf)
