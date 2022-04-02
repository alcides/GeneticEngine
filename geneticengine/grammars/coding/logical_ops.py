from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Callable

from geneticengine.grammars.coding.classes import Condition


@dataclass
class And(Condition):
    """attributes left and right"""

    left: Condition
    right: Condition

    def evaluate(self, **kwargs) -> bool:
        return self.left.evaluate(**kwargs) and self.right.evaluate(**kwargs)

    def evaluate_lines(self, **kwargs) -> Callable[[Any], bool]:
        return lambda line: self.left.evaluate_lines(**kwargs)(
            line,
        ) and self.right.evaluate_lines(**kwargs)(line)

    def __str__(self):
        return f"({self.left} and {self.right})"


@dataclass
class Or(Condition):
    """attributes left and right"""

    left: Condition
    right: Condition

    def evaluate(self, **kwargs) -> bool:
        return self.left.evaluate(**kwargs) or self.right.evaluate(**kwargs)

    def evaluate_lines(self, **kwargs) -> Callable[[Any], bool]:
        return lambda line: self.left.evaluate_lines(**kwargs)(
            line,
        ) or self.right.evaluate_lines(**kwargs)(line)

    def __str__(self):
        return f"({self.left} or {self.right})"


@dataclass
class Not(Condition):
    """attribute cond"""

    cond: Condition

    def evaluate(self, **kwargs) -> bool:
        return not self.cond.evaluate(**kwargs)

    def evaluate_lines(self, **kwargs) -> Callable[[Any], bool]:
        return lambda line: not self.cond.evaluate_lines(**kwargs)(line)

    def __str__(self):
        return f"(not {self.cond})"
