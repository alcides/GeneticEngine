"""Metamorphic-testing benchmark inspired by MTGP.

The target function is sine. Besides a few labeled examples, candidates are
checked against the metamorphic relation ``f(x) == f(x + 2*pi)``. This shows
how follow-up inputs can add behavioral constraints without additional labels.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Annotated

from examples.benchmarks.benchmark import Benchmark, example_run
from geneticengine.grammar.grammar import Grammar, extract_grammar
from geneticengine.grammar.metahandlers.floats import FloatRange
from geneticengine.problems import Problem, SingleObjectiveProblem


class ScalarExpression(ABC):
    @abstractmethod
    def evaluate(self, x: float) -> float: ...


@dataclass
class Input(ScalarExpression):
    def evaluate(self, x: float) -> float:
        return x


@dataclass
class Constant(ScalarExpression):
    value: Annotated[float, FloatRange(-3.0, 3.0)]

    def evaluate(self, x: float) -> float:
        return self.value


@dataclass
class Add(ScalarExpression):
    left: ScalarExpression
    right: ScalarExpression

    def evaluate(self, x: float) -> float:
        return self.left.evaluate(x) + self.right.evaluate(x)


@dataclass
class Multiply(ScalarExpression):
    left: ScalarExpression
    right: ScalarExpression

    def evaluate(self, x: float) -> float:
        return self.left.evaluate(x) * self.right.evaluate(x)


@dataclass
class Sine(ScalarExpression):
    argument: ScalarExpression

    def evaluate(self, x: float) -> float:
        return math.sin(self.argument.evaluate(x))


class MetamorphicTestingBenchmark(Benchmark):
    labeled_cases = ((0.0, 0.0), (math.pi / 2, 1.0), (math.pi, 0.0))
    base_inputs = (-2.3, -0.7, 0.4, 1.8, 3.1)

    def __init__(self):
        self.grammar = extract_grammar([Input, Constant, Add, Multiply, Sine], ScalarExpression)
        self.problem = SingleObjectiveProblem(minimize=True, target=0.0, fitness_function=self.fitness)

    def fitness(self, expression: ScalarExpression) -> float:
        labeled_error = sum((expression.evaluate(x) - y) ** 2 for x, y in self.labeled_cases)
        metamorphic_error = sum(
            abs(expression.evaluate(x) - expression.evaluate(x + 2 * math.pi))
            for x in self.base_inputs
        )
        return labeled_error + metamorphic_error

    def get_problem(self) -> Problem:
        return self.problem

    def get_grammar(self) -> Grammar:
        return self.grammar


if __name__ == "__main__":
    example_run(MetamorphicTestingBenchmark())
