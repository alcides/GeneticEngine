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
class SafeSqrt(Number):
    number: Number

    def keep_safe(self, x):
        return abs(x)

    def evaluate(self, **kwargs):
        return np.sqrt(self.keep_safe(self.number.evaluate(**kwargs)))
    
    def evaluate_lines(self, **kwargs) -> Callable[[Any], float]:
        return lambda line: np.sqrt(self.keep_safe(self.number.evaluate_lines(**kwargs)(line)))

    def __str__(self) -> str:
        return f"np.sqrt({self.number})"


@dataclass
class Sin(Number):
    number: Number

    def evaluate(self, **kwargs):
        return np.sin(self.number.evaluate(**kwargs))
    
    def evaluate_lines(self, **kwargs) -> Callable[[Any], float]:
        return lambda line: np.sin(self.number.evaluate_lines(**kwargs)(line))

    def __str__(self) -> str:
        return f"np.sin({self.number})"


@dataclass
class Tanh(Number):
    number: Number

    def evaluate(self, **kwargs):
        return np.tanh(self.number.evaluate(**kwargs))
    
    def evaluate_lines(self, **kwargs) -> Callable[[Any], float]:
        return lambda line: np.tanh(self.number.evaluate_lines(**kwargs)(line))

    def __str__(self) -> str:
        return f"np.tanh({self.number})"


@dataclass
class Exp(Number):
    number: Number
    
    def keep_safe(self, x):
        if isnan(x):
            x = 0
        x = max(min(10000,x),0)
        return x

    def evaluate(self, **kwargs):
        return np.exp(self.number.evaluate(**kwargs))
    
    def evaluate_lines(self, **kwargs) -> Callable[[Any], float]:
        return lambda line: np.exp(self.keep_safe(self.number.evaluate_lines(**kwargs)(line)))

    def __str__(self) -> str:
        return f"np.exp({self.number})"


@dataclass
class SafeLog(Number):
    number: Number
    
    def keep_safe(self, x):
        return 1 + abs(x)

    def evaluate(self, **kwargs):
        return np.log(self.keep_safe(self.number.evaluate(**kwargs)))
    
    def evaluate_lines(self, **kwargs) -> Callable[[Any], float]:
        return lambda line: np.log(self.keep_safe(self.number.evaluate_lines(**kwargs)(line)))

    def __str__(self) -> str:
        return f"np.log({self.number})"
