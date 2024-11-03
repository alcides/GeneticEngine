from __future__ import annotations


from examples.benchmarks.benchmark import Benchmark, example_run
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.grammar import Grammar
from geneticengine.problems import Problem
from geneticengine.problems import SingleObjectiveProblem


from dataclasses import dataclass

from geml.grammars.coding.classes import Expr
from geml.grammars.coding.classes import XAssign
from geml.grammars.coding.control_flow import Code
from geml.grammars.coding.control_flow import ForLoop


# ===================================
# This is a simple example on how to use GeneticEngine to solve a GP problem.
# We define the tree structure of the representation and then we define the fitness function for our problem
# The Pymax problem is a traditional maximisation problem, where the goal is to produce as large a number as possible.
# ===================================


class VarX(Expr):
    def evaluate(self, x=0):
        return x

    def __str__(self) -> str:
        return "x"


class Const(Expr):
    def evaluate(self, x=0):
        return 0.5

    def __str__(self) -> str:
        return "0.5"


@dataclass
class XPlusConst(Expr):
    right: Const

    def evaluate(self, x):
        return x + self.right.evaluate(x)

    def __str__(self) -> str:
        return f"x + {self.right}"


@dataclass
class XTimesConst(Expr):
    right: Const

    def evaluate(self, x):
        return x * self.right.evaluate(x)

    def __str__(self) -> str:
        return f"x * {self.right}"


class PyMaxBenchmark(Benchmark):
    def __init__(self):
        self.setup_problem()
        self.setup_grammar()

    def setup_problem(self):
        self.problem = SingleObjectiveProblem(minimize=False, fitness_function=lambda x: x.evaluate())

    def setup_grammar(self):
        self.grammar = extract_grammar(
            [XPlusConst, XTimesConst, XAssign, ForLoop, Code, Const, VarX],
            ForLoop,
        )

    def get_problem(self) -> Problem:
        return self.problem

    def get_grammar(self) -> Grammar:
        return self.grammar


if __name__ == "__main__":
    example_run(PyMaxBenchmark())
