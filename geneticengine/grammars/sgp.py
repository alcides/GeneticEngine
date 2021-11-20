from abc import ABC
from typing import Annotated, Any, Callable
from geneticengine.exceptions import GeneticEngineError
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.vars import VarRange


from typing import Annotated
from dataclasses import dataclass


class Number(ABC):
    def evaluate(self, **kwargs) -> float:
        return 0.0

    def evaluate_lines(self, **kwargs) -> Callable[[Any], float]:
        return lambda line: 0.0


@dataclass
class Plus(Number):
    left: Number
    right: Number

    def evaluate(self, **kwargs):
        return self.left.evaluate(**kwargs) + self.right.evaluate(**kwargs)

    def evaluate_lines(self, **kwargs):
        return lambda line: self.left.evaluate_lines(**kwargs)(line) + self.right.evaluate_lines(**kwargs)(line)

    def __str__(self) -> str:
        return f"({self.left} + {self.right})"


@dataclass
class Mul(Number):
    left: Number
    right: Number

    def evaluate(self, **kwargs):
        return self.left.evaluate(**kwargs) * self.right.evaluate(**kwargs)

    def evaluate_lines(self, **kwargs):
        return lambda line: self.left.evaluate_lines(**kwargs)(line) * self.right.evaluate_lines(**kwargs)(line)
    
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

    def evaluate_lines(self, **kwargs):
        d1 = lambda line: self.left.evaluate_lines(**kwargs)(line)
        d2 = lambda line: self.keep_safe(self.right.evaluate_lines(**kwargs)(line))
        return lambda line: d1(line) / d2(line)

    def __str__(self) -> str:
        return f"({self.left}/{self.right})"


@dataclass
class Literal(Number):
    val: Annotated[int, IntRange(-10, 11)]
    
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
        if not hasattr(self,"feature_indices"):
            raise GeneticEngineError("To use geneticengine.grammars.sgp.Var.evaluate_lines, one must specify a Var.feature_indices dictionary.")
        return lambda line: line[self.feature_indices[self.name]]

    def __str__(self) -> str:
        return self.name
