from abc import ABC
from dataclasses import dataclass
from typing import Annotated, Any, Callable

from geneticengine.core.decorators import abstract

    
class Statement(ABC):
    def evaluate(self, **kwargs) -> float:
        return 0

    def evaluate_lines(self, **kwargs) -> Callable[[Any], float]:
        return lambda _: 0
    
class Expr(ABC):
    def evaluate(self, **kwargs) -> float:
        return 0.0
    
    def evaluate_lines(self, **kwargs) -> Callable[[Any], float]:
        return lambda _: 0.0

    
@abstract
class Number(Expr):
    pass

@abstract
class Condition(Expr):
    pass

@dataclass
class XAssign(Statement):
    value: Expr

    def evaluate(self, x: float = 1) -> float:
        return self.value.evaluate(x)
    
    def evaluate_lines(self, **kwargs) -> Callable[[Any], float]:
        return lambda line: self.value.evaluate_lines(**kwargs)(line)

    def __str__(self):
        return "x = {}".format(self.value)
