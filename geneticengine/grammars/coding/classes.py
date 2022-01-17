from abc import ABC
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Union

from geneticengine.core.decorators import abstract


class Statement(ABC):
    def evaluate(self, **kwargs) -> float:
        return 0

    def evaluate_lines(self, **kwargs) -> Callable[[Any], float]:
        return lambda _: 0


class Expr(ABC):
    def evaluate(self, **kwargs) -> Union[float, bool]:
        return 0.0

    def evaluate_lines(self, **kwargs) -> Callable[[Any], float]:
        return lambda _: 0.0


@abstract
class Number(Expr):
    pass


@abstract
class Condition(Expr):
    pass


@abstract
class NumberList(Expr):
    pass


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
        return "x = {}".format(self.value)
