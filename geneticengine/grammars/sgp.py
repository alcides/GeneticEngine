from abc import ABC
from geneticengine.core.tree import TreeNode
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.vars import VarRange


from typing import Annotated
from dataclasses import dataclass


class Number(ABC):
    def evaluate(self, **kwargs) -> float:
        return 0.0


@dataclass
class Plus(Number):
    left: Number
    right: Number

    def evaluate(self, **kwargs):
        return self.left.evaluate(**kwargs) + self.right.evaluate(**kwargs)

    def __str__(self) -> str:
        return f"({self.left} + {self.right})"


@dataclass
class Mul(Number):
    left: Number
    right: Number

    def evaluate(self, **kwargs):
        return self.left.evaluate(**kwargs) * self.right.evaluate(**kwargs)

    def __str__(self) -> str:
        return f"({self.left} * {self.right})"


@dataclass
class SafeDiv(Number):
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
class Literal(Number):
    val: Annotated[int, IntRange(-10, 11)]

    def evaluate(self, **kwargs):
        return self.val

    def __str__(self) -> str:
        return str(self.val)


@dataclass
class Var(Number):
    name: Annotated[str, VarRange(["x", "y", "z"])]

    def evaluate(self, **kwargs):
        return kwargs[self.name]

    def __str__(self) -> str:
        return self.name
