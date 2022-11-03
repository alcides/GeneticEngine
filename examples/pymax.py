from __future__ import annotations

from dataclasses import dataclass

from geneticengine.algorithms.gp.gp_friendly import GPFriendly
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.representations.grammatical_evolution.dynamic_structured_ge import (
    DynamicStructureGrammaticalEvolutionRepresentation,
)
from geneticengine.core.representations.grammatical_evolution.ge import (
    GrammaticalEvolutionRepresentation,
)
from geneticengine.core.representations.grammatical_evolution.structured_ge import (
    StructureGrammaticalEvolutionRepresentation,
)
from geneticengine.core.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.grammars.coding.classes import Expr
from geneticengine.grammars.coding.classes import XAssign
from geneticengine.grammars.coding.control_flow import Code
from geneticengine.grammars.coding.control_flow import ForLoop

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


def fit(indiv: Code):
    return indiv.evaluate()


def fitness_function(x):
    return fit(x)


def preprocess():
    return extract_grammar(
        [XPlusConst, XTimesConst, XAssign, ForLoop, Code, Const, VarX],
        ForLoop,
    )


def evolve(
    g,
    seed,
    mode,
    representation="TreeBasedRepresentation",
):
    if representation == "ge":
        representation = GrammaticalEvolutionRepresentation
    elif representation == "sge":
        representation = GrammaticalEvolutionRepresentation
    elif representation == "dsge":
        representation = GrammaticalEvolutionRepresentation
    else:
        representation = TreeBasedRepresentation
    alg = GPFriendly(
        g,
        representation=representation,
        problem=SingleObjectiveProblem(
            minimize=False,
            fitness_function=fitness_function,
            target_fitness=None,
        ),
        max_depth=8,
        population_size=25,
        number_of_generations=10,
        seed=seed,
        timer_stop_criteria=mode,
    )
    (b, bf, bp) = alg.evolve()
    return b, bf


if __name__ == "__main__":
    g = preprocess()
    bf, b = evolve(g, 0, False)
    print(b)
    print(f"With fitness: {bf}")
