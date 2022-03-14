from abc import ABC
from typing import Annotated

from geneticengine.core.utils import fdataclass
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.vars import VarRange


class Number(ABC):
    def evaluate(self, **kwargs) -> float:
        return 0.0


@fdataclass
class Plus(Number):
    left: Number
    right: Number

    def evaluate(self, **kwargs):
        return self.left.evaluate(**kwargs) + self.right.evaluate(**kwargs)

    def __str__(self) -> str:
        return f"({self.left} + {self.right})"


@fdataclass
class Minus(Number):
    left: Number
    right: Number

    def evaluate(self, **kwargs):
        return self.left.evaluate(**kwargs) - self.right.evaluate(**kwargs)

    def __str__(self) -> str:
        return f"({self.left} - {self.right})"


@fdataclass
class Mul(Number):
    left: Number
    right: Number

    def evaluate(self, **kwargs):
        return self.left.evaluate(**kwargs) * self.right.evaluate(**kwargs)

    def __str__(self) -> str:
        return f"({self.left} * {self.right})"


@fdataclass
class SafeDiv(Number):
    left: Number
    right: Number

    def evaluate(self, **kwargs):
        d1 = self.left.evaluate(**kwargs)
        d2 = self.right.evaluate(**kwargs)
        if d2 == 0:
            return 0.00001
        return d1 / d2

    def __str__(self) -> str:
        return f"({self.left}/{self.right})"


@fdataclass
class Literal(Number):
    val: Annotated[int, IntRange(0, 9)]

    def evaluate(self, **kwargs):
        return self.val

    def __str__(self) -> str:
        return str(self.val)


@fdataclass
class Var(Number):
    name: Annotated[str, VarRange(["x", "y", "z"])]

    def evaluate(self, **kwargs):
        return kwargs[self.name]

    def __str__(self) -> str:
        return self.name
