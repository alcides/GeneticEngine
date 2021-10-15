from geneticengine.core.tree import Node
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.vars import VarRange


from typing import Annotated
from dataclasses import dataclass


class Number(object):
    pass


@dataclass
class Plus(Node, Number):
    left: Number
    right: Number

    def evaluate(self, **kwargs):
        return self.left.evaluate(**kwargs) + self.right.evaluate(**kwargs)

    def __str__(self) -> str:
        return f"({self.left} + {self.right})"


@dataclass
class Mul(Node, Number):
    left: Number
    right: Number

    def evaluate(self, **kwargs):
        return self.left.evaluate(**kwargs) * self.right.evaluate(**kwargs)

    def __str__(self) -> str:
        return f"({self.left} * {self.right})"


@dataclass
class SafeDiv(Node, Number):
    left: Number
    right: Number

    def evaluate(self, **kwargs):
        d1 = self.left.evaluate(**kwargs)
        d2 = self.right.evaluate(**kwargs)
        if d2 == 0:
            d2 = 0.000001
        return d1 / d2

    def __str__(self) -> str:
        return f"({self.left}/{self.right})"


@dataclass
class Literal(Node, Number):
    val: Annotated[int, IntRange(-10, 11)]

    def evaluate(self, **kwargs):
        return self.val

    def __str__(self) -> str:
        return str(self.val)


@dataclass
class Var(Node, Number):
    val: Annotated[str, VarRange(["x", "y", "z"])]

    def evaluate(self, **kwargs):
        return kwargs[self.val]

    def __str__(self) -> str:
        return self.val
