from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Annotated

from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.vars import VarRange


class Number(ABC):
    """Standard Number object."""

    def evaluate(self, **kwargs):
        return 0.0


@dataclass
class Plus(Number):
    """Standard Plus object.

    Args:
        left  (Number)
        right (Number)

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
    """Standard Minus object.

    Args:
        left  (Number)
        right (Number)

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
    """Standard Multiplication object.

    Args:
        left  (Number)
        right (Number)

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
    """Standard Literal object.

    Args:
        val  (Number)

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
    """Standard Variable object. Used to introduce variables.

    Args:
        name  (str)

    Returns when evaluated:
        name
    """

    name: Annotated[str, VarRange(["x", "y", "z"])]

    def evaluate(self, **kwargs):
        return kwargs[self.name]

    def __str__(self) -> str:
        return self.name


def simplify(x: Number) -> Number:
    if isinstance(x, Plus):
        l = simplify(x.left)
        r = simplify(x.right)
        if isinstance(l, Literal) and isinstance(r, Literal):
            return Literal(l.val + r.val)
        elif isinstance(r, Literal):
            l, r = r, l

        if isinstance(l, Literal) and l.val == 0:
            return r

        if isinstance(l, Literal) and isinstance(r, Plus) and isinstance(r.left, Literal):
            return simplify(Plus(Plus(l, r.left), r.right))

        if isinstance(l, Plus):
            return Plus(l.left, Plus(l.right, r))

        return Plus(l, r)
    elif isinstance(x, Mul):
        l = simplify(x.left)
        r = simplify(x.right)
        if isinstance(l, Literal) and isinstance(r, Literal):
            return Literal(l.val * r.val)
        elif isinstance(r, Literal):
            l, r = r, l

        if isinstance(l, Literal) and l.val == 1:
            return r

        if isinstance(l, Literal) and l.val == 0:
            return l

        if isinstance(r, Mul) and isinstance(r.left, Literal):
            return simplify(Mul(Mul(l, r.left), r.right))

        if isinstance(l, Mul):
            return Mul(l.left, Mul(l.right, r))

        return Plus(l, r)

    else:
        return x
