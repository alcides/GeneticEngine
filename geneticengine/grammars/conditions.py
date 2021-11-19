from abc import ABC
from dataclasses import dataclass
from typing import Annotated, List
from geneticengine.metahandlers.ints import IntRange


class Condition(ABC):
    def evaluate(self, x: bool) -> bool:
        return False
    
class Expr(ABC):
    def evaluate(self, x: float) -> float:
        return 0.0

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
class SmallerThan(Condition):
    left: Expr
    right: Expr
    
    def evaluate(self, x: float = 1) -> bool:
        return self.left.evaluate(x) < self.right.evaluate(x)

    def __str__(self):
        return f"({self.left} < {self.right})"

@dataclass
class SmallerOrEqualThan(Condition):
    left: Expr
    right: Expr
    
    def evaluate(self, x: float = 1) -> bool:
        return self.left.evaluate(x) <= self.right.evaluate(x)

    def __str__(self):
        return f"({self.left} <= {self.right})"

