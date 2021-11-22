from abc import ABC
from typing import Annotated, Any, Callable


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


class Condition(ABC):
    def evaluate(self, **kwargs) -> bool:
        return False

    def evaluate_lines(self, **kwargs) -> Callable[[Any], bool]:
        return lambda _: False
