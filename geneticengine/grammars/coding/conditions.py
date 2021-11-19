from abc import ABC
from dataclasses import dataclass
from typing import Annotated, List
from geneticengine.metahandlers.ints import IntRange
from geneticengine.grammars.coding.classes import Expr, Condition


@dataclass
class Equals(Condition):
    left: Expr
    right: Expr
    
    def evaluate(self, x: float = 1) -> bool:
        return self.left.evaluate(x) == self.right.evaluate(x)

    def __str__(self):
        return f"({self.left} == {self.right})"

@dataclass
class NotEquals(Condition):
    left: Expr
    right: Expr
    
    def evaluate(self, x: float = 1) -> bool:
        return self.left.evaluate(x) != self.right.evaluate(x)

    def __str__(self):
        return f"({self.left} != {self.right})"

@dataclass
class GreaterThan(Condition):
    left: Expr
    right: Expr
    
    def evaluate(self, x: float = 1) -> bool:
        return self.left.evaluate(x) > self.right.evaluate(x)

    def __str__(self):
        return f"({self.left} > {self.right})"

@dataclass
class GreaterOrEqualThan(Condition):
    left: Expr
    right: Expr
    
    def evaluate(self, x: float = 1) -> bool:
        return self.left.evaluate(x) >= self.right.evaluate(x)

    def __str__(self):
        return f"({self.left} >= {self.right})"

@dataclass
class LessThan(Condition):
    left: Expr
    right: Expr
    
    def evaluate(self, x: float = 1) -> bool:
        return self.left.evaluate(x) < self.right.evaluate(x)

    def __str__(self):
        return f"({self.left} < {self.right})"

@dataclass
class LessOrEqualThan(Condition):
    left: Expr
    right: Expr
    
    def evaluate(self, x: float = 1) -> bool:
        return self.left.evaluate(x) <= self.right.evaluate(x)

    def __str__(self):
        return f"({self.left} <= {self.right})"


@dataclass
class Is(Condition):
    left: Expr
    right: Expr
    
    def evaluate(self, x: float = 1) -> bool:
        return self.left.evaluate(x) is self.right.evaluate(x)

    def __str__(self):
        return f"({self.left} is {self.right})"

@dataclass
class IsNot(Condition):
    left: Expr
    right: Expr
    
    def evaluate(self, x: float = 1) -> bool:
        return self.left.evaluate(x) is not self.right.evaluate(x)

    def __str__(self):
        return f"({self.left} is not {self.right})"
