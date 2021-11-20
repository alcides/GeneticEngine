from abc import ABC
from dataclasses import dataclass
from textwrap import indent
from typing import Annotated, List
from geneticengine.grammars.coding.classes import Expr, Statement
from geneticengine.metahandlers.vars import VarRange
from geneticengine.metahandlers.ints import IntRange


@dataclass
class Plus(Expr):
    left: Expr
    right: Expr

    def evaluate(self, **kwargs):
        return self.left.evaluate(**kwargs) + self.right.evaluate(**kwargs)

    def __str__(self) -> str:
        return f"({self.left} + {self.right})"


@dataclass
class Mul(Expr):
    left: Expr
    right: Expr

    def evaluate(self, **kwargs):
        return self.left.evaluate(**kwargs) * self.right.evaluate(**kwargs)

    def __str__(self) -> str:
        return f"({self.left} * {self.right})"


@dataclass
class SafeDiv(Expr):
    left: Expr
    right: Expr

    def evaluate(self, **kwargs):
        d1 = self.left.evaluate(**kwargs)
        d2 = self.right.evaluate(**kwargs)
        if d2 == 0:
            d2 = 0.000001
        return d1 / d2

    def __str__(self) -> str:
        return f"({self.left}/{self.right})"


@dataclass
class Literal(Expr):
    val: Annotated[int, IntRange(-10, 11)]

    def evaluate(self, **kwargs):
        return self.val

    def __str__(self) -> str:
        return str(self.val)


@dataclass
class Var(Expr):
    name: Annotated[str, VarRange(["x", "y", "z"])]

    def evaluate(self, **kwargs):
        return kwargs[self.name]

    def __str__(self) -> str:
        return self.name
    
@dataclass
class XAssign(Statement):
    value: Expr

    def evaluate(self, x: float = 1.0) -> float:
        return self.value.evaluate(x)

    def __str__(self):
        return "x = {}".format(self.value)
    