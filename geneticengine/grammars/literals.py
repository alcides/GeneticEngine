from __future__ import annotations

from dataclasses import dataclass

from geneticengine.core.decorators import abstract
from geneticengine.grammars.sgp import Number


@abstract
class ExpLiteral(Number):
    pass


@dataclass
class One(ExpLiteral):
    def evaluate(self, **kwargs):
        return 1

    def __str__(self) -> str:
        return str(1)


@dataclass
class PointOne(ExpLiteral):
    def evaluate(self, **kwargs):
        return 0.1

    def __str__(self) -> str:
        return str(0.1)


@dataclass
class PointtOne(ExpLiteral):
    def evaluate(self, **kwargs):
        return 0.01

    def __str__(self) -> str:
        return str(0.01)


@dataclass
class PointttOne(ExpLiteral):
    def evaluate(self, **kwargs):
        return 0.001

    def __str__(self) -> str:
        return str(0.001)


@dataclass
class MinusPointttOne(ExpLiteral):
    def evaluate(self, **kwargs):
        return -0.001

    def __str__(self) -> str:
        return str(-0.001)


@dataclass
class MinusPointtOne(ExpLiteral):
    def evaluate(self, **kwargs):
        return -0.01

    def __str__(self) -> str:
        return str(-0.01)


@dataclass
class MinusPointOne(ExpLiteral):
    def evaluate(self, **kwargs):
        return -0.1

    def __str__(self) -> str:
        return str(-0.1)


@dataclass
class MinusOne(ExpLiteral):
    def evaluate(self, **kwargs):
        return -1

    def __str__(self) -> str:
        return str(-1)


exp_literals = [
    MinusOne,
    MinusPointOne,
    MinusPointtOne,
    MinusPointttOne,
    One,
    PointOne,
    PointtOne,
    PointttOne,
]
