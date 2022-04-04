from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated
from typing import Any
from typing import Callable

from geneticengine.exceptions import GeneticEngineError
from geneticengine.grammars.coding.classes import Expr
from geneticengine.grammars.coding.classes import Number
from geneticengine.grammars.coding.classes import Statement
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.vars import VarRange


@dataclass
class Max(Number):
    left: Number
    right: Number

    def evaluate(self, **kwargs):
        return max(self.left.evaluate(**kwargs), self.right.evaluate(**kwargs))

    def evaluate_lines(self, **kwargs) -> Callable[[Any], float]:
        return lambda line: max(
            self.left.evaluate_lines(**kwargs)(line),
            self.right.evaluate_lines(**kwargs)(line),
        )

    def __str__(self) -> str:
        return f"max({self.left},{self.right})"


@dataclass
class Min(Number):
    left: Number
    right: Number

    def evaluate(self, **kwargs):
        return min(self.left.evaluate(**kwargs), self.right.evaluate(**kwargs))

    def evaluate_lines(self, **kwargs) -> Callable[[Any], float]:
        return lambda line: min(
            self.left.evaluate_lines(**kwargs)(line),
            self.right.evaluate_lines(**kwargs)(line),
        )

    def __str__(self) -> str:
        return f"min({self.left},{self.right})"


@dataclass
class Abs(Number):
    value: Number

    def evaluate(self, **kwargs):
        return abs(self.value.evaluate(**kwargs))

    def evaluate_lines(self, **kwargs) -> Callable[[Any], float]:
        return lambda line: abs(self.value.evaluate_lines(**kwargs)(line))

    def __str__(self) -> str:
        return f"abs({self.value})"


@dataclass
class Plus(Number):
    left: Number
    right: Number

    def evaluate(self, **kwargs):
        return self.left.evaluate(**kwargs) + self.right.evaluate(**kwargs)

    def evaluate_lines(self, **kwargs) -> Callable[[Any], float]:
        return lambda line: self.left.evaluate_lines(**kwargs)(
            line,
        ) + self.right.evaluate_lines(**kwargs)(line)

    def __str__(self) -> str:
        return f"({self.left} + {self.right})"


@dataclass
class Mul(Number):
    left: Number
    right: Number

    def evaluate(self, **kwargs):
        return self.left.evaluate(**kwargs) * self.right.evaluate(**kwargs)

    def evaluate_lines(self, **kwargs) -> Callable[[Any], float]:
        return lambda line: self.left.evaluate_lines(**kwargs)(
            line,
        ) * self.right.evaluate_lines(**kwargs)(line)

    def __str__(self) -> str:
        return f"({self.left} * {self.right})"


@dataclass
class SafeDiv(Number):
    left: Number
    right: Number

    def keep_safe(self, d2):
        if d2 == 0:
            d2 = 0.000001
        return d2

    def evaluate(self, **kwargs):
        d1 = self.left.evaluate(**kwargs)
        d2 = self.right.evaluate(**kwargs)
        return d1 / self.keep_safe(d2)

    def evaluate_lines(self, **kwargs) -> Callable[[Any], float]:
        def d1(line):
            return self.left.evaluate_lines(**kwargs)(line)

        def d2(line):
            return self.keep_safe(
                self.right.evaluate_lines(**kwargs)(line),
            )

        return lambda line: d1(line) / d2(line)

    def __str__(self) -> str:
        return f"({self.left}/{self.right})"


@dataclass
class Literal(Number):
    val: Annotated[int, IntRange(0, 9)]

    def evaluate(self, **kwargs):
        return self.val

    def evaluate_lines(self, **kwargs):
        return lambda _: self.val

    def __str__(self) -> str:
        return str(self.val)


@dataclass
class Var(Number):
    name: Annotated[str, VarRange(["x", "y", "z"])]

    def evaluate(self, **kwargs):
        return kwargs[self.name]

    def evaluate_lines(self, **kwargs):
        if not hasattr(self, "feature_indices"):
            raise GeneticEngineError(
                "To use geneticengine.grammars.coding.expressions.Var.evaluate_lines, one must specify a Var.feature_indices dictionary.",
            )
        return lambda line: line[self.feature_indices[self.name]]

    def __str__(self) -> str:
        return self.name


@dataclass
class XAssign(Statement):
    value: Expr

    def evaluate(self, **kwargs) -> float:
        if "x" not in kwargs:
            kwargs["x"] = 1.0
        return self.value.evaluate(**kwargs)

    def evaluate_lines(self, **kwargs) -> Callable[[Any], float]:
        return lambda line: self.value.evaluate_lines(**kwargs)(line)

    def __str__(self):
        return f"x = {self.value}"


all_operators = [Max, Min, Abs, Plus, Mul, SafeDiv, Literal, Var, XAssign]
