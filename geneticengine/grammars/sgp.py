from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Annotated

from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.vars import VarRange


class Number(ABC):
    """
    Standard Number object.
    """

    def evaluate(self, **kwargs):
        return 0.0


@dataclass
class Plus(Number):
    """
    Standard Plus object.

    Parameters:
        - left  (Number)
        - right (Number)

    Returns when evaluated:
        left + right
    """

    left: Number
    right: Number

    def evaluate(self, **kwargs):
        return self.left.evaluate(**kwargs) + self.right.evaluate(**kwargs)

    def __str__(self) -> str:
        return f"({self.left} + {self.right})"


@dataclass
class Minus(Number):
    """
    Standard Minus object.

    Parameters:
        - left  (Number)
        - right (Number)

    Returns when evaluated:
        left - right
    """

    left: Number
    right: Number

    def evaluate(self, **kwargs):
        return self.left.evaluate(**kwargs) - self.right.evaluate(**kwargs)

    def __str__(self) -> str:
        return f"({self.left} - {self.right})"


@dataclass
class Mul(Number):
    """
    Standard Multiplication object.

    Parameters:
        - left  (Number)
        - right (Number)

    Returns when evaluated:
        left * right
    """

    left: Number
    right: Number

    def evaluate(self, **kwargs):
        return self.left.evaluate(**kwargs) * self.right.evaluate(**kwargs)

    def __str__(self) -> str:
        return f"({self.left} * {self.right})"



@dataclass
class Literal(Number):
    """
    Standard Literal object.

    Parameters:
        - val  (Number)

    Returns when evaluated:
        val
    """

    val: Annotated[int, IntRange(0, 9)]

    def evaluate(self, **kwargs):
        return self.val

    def __str__(self) -> str:
        return str(self.val)


@dataclass
class Var(Number):
    """
    Standard Variable object. Used to introduce variables.

    Parameters:
        - name  (str)

    Returns when evaluated:
        name
    """

    name: Annotated[str, VarRange(["x", "y", "z"])]

    def evaluate(self, **kwargs):
        return kwargs[self.name]

    def __str__(self) -> str:
        return self.name
