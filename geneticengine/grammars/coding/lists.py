from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Annotated
from typing import Any
from typing import Callable
from typing import List

from geneticengine.exceptions import GeneticEngineError
from geneticengine.grammars.coding.classes import Number
from geneticengine.grammars.coding.classes import NumberList
from geneticengine.metahandlers.lists import ListSizeBetween
from geneticengine.metahandlers.vars import VarRange


@dataclass
class Max(Number):
    list: NumberList

    def evaluate(self, **kwargs):
        return max(self.list.evaluate(**kwargs))

    def evaluate_lines(self, **kwargs) -> Callable[[Any], float]:
        return lambda line: max(self.list.evaluate_lines(**kwargs)(line))

    def __str__(self) -> str:
        return f"max({self.list})"


@dataclass
class Min(Number):
    list: NumberList

    def evaluate(self, **kwargs):
        return min(self.list.evaluate(**kwargs))

    def evaluate_lines(self, **kwargs) -> Callable[[Any], float]:
        return lambda line: min(self.list.evaluate_lines(**kwargs)(line))

    def __str__(self) -> str:
        return f"min({self.list})"


@dataclass
class Length(Number):
    list: NumberList

    def evaluate(self, **kwargs):
        return len(self.list.evaluate(**kwargs))

    def evaluate_lines(self, **kwargs) -> Callable[[Any], float]:
        return lambda line: len(self.list.evaluate_lines(**kwargs)(line))

    def __str__(self) -> str:
        return f"len({self.list})"


@dataclass
class Combine(NumberList):
    list1: NumberList
    list2: NumberList

    def evaluate(self, **kwargs):
        return self.list1.evaluate(**kwargs) + self.list2.evaluate(**kwargs)

    def evaluate_lines(self, **kwargs) -> Callable[[Any], list[float]]:
        return lambda line: self.list1.evaluate_lines(**kwargs)(
            line,
        ) + self.list2.evaluate_lines(**kwargs)(line)

    def __str__(self) -> str:
        return f"({self.list1} + {self.list2})"


@dataclass
class Literal(NumberList):
    list: Annotated[list[Number], ListSizeBetween(2, 3)]

    def evaluate(self, **kwargs):
        return [v.evaluate(**kwargs) for v in self.list]

    def evaluate_lines(self, **kwargs):
        return lambda line: [v.evaluate_lines(**kwargs)(line) for v in self.list]

    def __str__(self) -> str:
        return str(self.list)


@dataclass
class GetElement(Number):
    list: NumberList
    element: Number

    def evaluate(self, **kwargs):
        list_length = Length(self.list).evaluate(**kwargs)
        return self.list.evaluate(**kwargs)[
            round(self.element.evaluate(**kwargs)) % list_length
        ]

    def evaluate_lines(self, **kwargs):
        def list_length(line):
            return Length(self.list).evaluate_lines(
                **kwargs,
            )(line)

        return lambda line: self.list.evaluate_lines(**kwargs)(line)[
            round(
                self.element.evaluate_lines(
                    **kwargs,
                )(line),
            )
            % list_length(line)
        ]

    def __str__(self) -> str:
        return f"{self.list}[{self.element}]"


@dataclass
class Var(NumberList):
    name: Annotated[str, VarRange(["x", "y", "z"])]

    def evaluate(self, **kwargs):
        return kwargs[self.name]

    def evaluate_lines(self, **kwargs):
        if not hasattr(self, "feature_indices"):
            raise GeneticEngineError(
                "To use geneticengine.grammars.coding.lists.Var.evaluate_lines, one must specify a Var.feature_indices dictionary.",
            )
        return lambda line: line[self.feature_indices[self.name]]

    def __str__(self) -> str:
        return self.name


# import geneticengine.grammars.coding.numbers as numbers
# import IPython as ip
# ip.embed()
