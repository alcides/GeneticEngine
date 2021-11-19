from dataclasses import dataclass
from textwrap import indent
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.treebased import treebased_representation
from geneticengine.algorithms.gp.gp import GP
from geneticengine.grammars.control_flow import Statement, Code, ForLoop
from geneticengine.grammars.conditions import Expr


@dataclass
class XAssign(Statement):
    value: Expr

    def evaluate(self, x: float = 1.0) -> float:
        return self.value.evaluate(x)

    def __str__(self):
        return "x = {}".format(self.value)

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
        return "x + {}".format(self.right)


@dataclass
class XTimesConst(Expr):
    right: Const

    def evaluate(self, x):
        return x * self.right.evaluate(x)

    def __str__(self) -> str:
        return "x * {}".format(self.right)


def fit(indiv: Code):
    return indiv.evaluate(0.0)


fitness_function = lambda x: fit(x)

if __name__ == "__main__":
    g = extract_grammar(
        [XPlusConst, XTimesConst, XAssign, ForLoop, Code, Const, VarX], Code
    )
    alg = GP(
        g,
        treebased_representation,
        fitness_function,
        max_depth=10,
        population_size=40,
        number_of_generations=15,
        minimize=False,
    )
    (b, bf, bp) = alg.evolve(verbose=0)
    print(bp, b)
    print("With fitness: {}".format(bf))
