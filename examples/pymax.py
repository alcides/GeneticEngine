from __future__ import annotations

from dataclasses import dataclass

from geneticengine.algorithms.gp.simplegp import SimpleGP
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.grammar import Grammar
from geneticengine.core.problems import Problem
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.grammars.coding.classes import Expr
from geneticengine.grammars.coding.classes import XAssign
from geneticengine.grammars.coding.control_flow import Code
from geneticengine.grammars.coding.control_flow import ForLoop

# ===================================
# This is a simple example on how to use GeneticEngine to solve a GP problem.
# We define the tree structure of the representation and then we define the fitness function for our problem
# The Pymax problem is a traditional maximisation problem, where the goal is to produce as large a number as possible.
# ===================================

# TODO: This example is not running correctly


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


def fit(indiv: Code):
    return indiv.evaluate()


class PyMaxBenchmark:
    def get_problem(self) -> Problem:
        return SingleObjectiveProblem(
            minimize=False,
            fitness_function=fit,
            target_fitness=None,
        )

    def get_grammar(self) -> Grammar:
        return extract_grammar(
            [XPlusConst, XTimesConst, XAssign, ForLoop, Code, Const, VarX],
            ForLoop,
        )

    def main(self, **args):
        g = self.get_grammar()
        prob = self.get_problem()
        alg = SimpleGP(g, problem=prob, max_depth=8, population_size=25, number_of_generations=10, **args)
        best = alg.evolve()
        fitness = prob.overall_fitness(best.get_phenotype())
        print(f"Fitness of {fitness} by genotype: {best.genotype} with phenotype: {best.get_phenotype()}")


if __name__ == "__main__":
    PyMaxBenchmark().main(seed=0)
