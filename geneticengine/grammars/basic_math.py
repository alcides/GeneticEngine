from abc import ABC
from typing import Callable, Any
from geneticengine.core.tree import TreeNode
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.vars import VarRange

from typing import Annotated
from dataclasses import dataclass

import numpy as np
from math import isnan

from geneticengine.grammars.sgp import Number


@dataclass
class SafeDiv(Number):
    left: Number
    right: Number

    def evaluate(self, **kwargs):
        x = self.left.evaluate(**kwargs)
        y = self.right.evaluate(**kwargs)
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(y == 0, np.ones_like(x), x / y)
        except ZeroDivisionError:
            # In this case we are trying to divide two constants, one of which is 0
            # Return a constant.
            return 1.0

    def __str__(self) -> str:
        return f"({self.left}/{self.right})"


@dataclass
class SafeSqrt(Number):
    number: Number

    def evaluate(self, **kwargs):
        v = self.number.evaluate(**kwargs)
        return np.sqrt(np.abs(v))

    def __str__(self) -> str:
        return f"np.sqrt(np.abs({self.number}))"


@dataclass
class Sin(Number):
    number: Number

    def evaluate(self, **kwargs):
        v = self.number.evaluate(**kwargs)
        return np.sin(v)

    def __str__(self) -> str:
        return f"np.sin({self.number})"


@dataclass
class Tanh(Number):
    number: Number

    def evaluate(self, **kwargs):
        v = self.number.evaluate(**kwargs)
        return np.tanh(v)

    def __str__(self) -> str:
        return f"np.tanh({self.number})"


@dataclass
class Exp(Number):
    number: Number

    def evaluate(self, **kwargs):
        v = self.number.evaluate(**kwargs)
        return np.exp(v)

    def __str__(self) -> str:
        return f"np.exp({self.number})"


@dataclass
class SafeLog(Number):
    number: Number

    def evaluate(self, **kwargs):
        v = self.number.evaluate(**kwargs)
        return np.log(1 + np.abs(v))

    def __str__(self) -> str:
        return f"np.log(1 + np.abs({self.number}))"
