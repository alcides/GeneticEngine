"""Dependency-free neural-network benchmarks inspired by Neuroevolution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated

from examples.benchmarks.benchmark import Benchmark, example_run
from geneticengine.grammar.grammar import Grammar, extract_grammar
from geneticengine.grammar.metahandlers.floats import FloatRange
from geneticengine.grammar.metahandlers.lists import ListSizeBetween
from geneticengine.problems import Problem, SingleObjectiveProblem

WEIGHTS = Annotated[float, FloatRange(-5.0, 5.0)]


@dataclass
class NeuralNetwork:
    """A fixed 2-2-1 tanh network encoded as nine evolvable weights."""

    weights: Annotated[list[WEIGHTS], ListSizeBetween(9, 9)]

    def predict(self, first: float, second: float) -> float:
        w = self.weights
        hidden_0 = math.tanh(w[0] * first + w[1] * second + w[2])
        hidden_1 = math.tanh(w[3] * first + w[4] * second + w[5])
        return math.tanh(w[6] * hidden_0 + w[7] * hidden_1 + w[8])


XOR_CASES = ((0.0, 0.0, -1.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, -1.0))
AND_CASES = ((0.0, 0.0, -1.0), (0.0, 1.0, -1.0), (1.0, 0.0, -1.0), (1.0, 1.0, 1.0))


def _fitness(network: NeuralNetwork, cases: tuple[tuple[float, float, float], ...]) -> float:
    return sum((network.predict(left, right) - expected) ** 2 for left, right, expected in cases) / len(cases)


def xor_fitness(network: NeuralNetwork) -> float:
    return _fitness(network, XOR_CASES)


def and_fitness(network: NeuralNetwork) -> float:
    return _fitness(network, AND_CASES)


@dataclass
class XorBenchmark(Benchmark):
    problem: Problem = SingleObjectiveProblem(xor_fitness, minimize=True, target=0)
    grammar: Grammar | None = None

    def __post_init__(self):
        self.grammar = extract_grammar([NeuralNetwork], NeuralNetwork)

    def get_problem(self) -> Problem:
        return self.problem

    def get_grammar(self) -> Grammar:
        assert self.grammar is not None
        return self.grammar


@dataclass
class AndBenchmark(XorBenchmark):
    def __post_init__(self):
        super().__post_init__()
        self.problem = SingleObjectiveProblem(and_fitness, minimize=True, target=0)


if __name__ == "__main__":
    example_run(XorBenchmark())
