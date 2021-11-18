from abc import ABC
from geneticengine.core.tree import TreeNode
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.vars import VarRange

from typing import Annotated
from dataclasses import dataclass

import numpy as np

from geneticengine.grammars.sgp import Number


@dataclass
class Sqrt(Number):
    number: Number

    def evaluate(self, **kwargs):
        return np.sqrt(self.number.evaluate(**kwargs))

    def __str__(self) -> str:
        return f"np.sqrt{self.number}"


@dataclass
class Sin(Number):
    number: Number

    def evaluate(self, **kwargs):
        return np.sin(self.number.evaluate(**kwargs))

    def __str__(self) -> str:
        return f"np.sin({self.number})"


@dataclass
class Tanh(Number):
    number: Number

    def evaluate(self, **kwargs):
        return np.tanh(self.number.evaluate(**kwargs))

    def __str__(self) -> str:
        return f"np.tanh({self.number})"


@dataclass
class Exp(Number):
    number: Number

    def evaluate(self, **kwargs):
        return np.exp(self.number.evaluate(**kwargs))

    def __str__(self) -> str:
        return f"np.exp({self.number})"


@dataclass
class Log(Number):
    number: Number

    def evaluate(self, **kwargs):
        return np.log(self.number.evaluate(**kwargs))

    def __str__(self) -> str:
        return f"np.log({self.number})"
