from abc import ABC
from dataclasses import dataclass
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.treebased import treebased_representation
from geneticengine.core.tree import PrettyPrintable
from geneticengine.metahandlers.lists import ListSizeBetween
from geneticengine.algorithms.gp.gp import GP


class Expr(ABC, PrettyPrintable):
    def evaluate(self, **kwargs) -> float:
        return 0


class One(Expr):
    def evaluate(self, **kwargs) -> float:
        return 1

    def __str__(self) -> str:
        return "1"


@dataclass
class Add(Expr):
    left: Expr
    right: Expr

    def evaluate(self, **kwargs):
        return self.left.evaluate() + self.right.evaluate()

    def __str__(self) -> str:
        return "(" + str(self.left) + " + " + str(self.right) + ")"


@dataclass
class Sub(Expr):
    left: Expr
    right: Expr

    def evaluate(self, **kwargs):
        return self.left.evaluate() - self.right.evaluate()

    def __str__(self) -> str:
        return "(" + str(self.left) + " - " + str(self.right) + ")"


@dataclass
class Mul(Expr):
    left: Expr
    right: Expr

    def evaluate(self, **kwargs) -> float:
        return self.left.evaluate() * self.right.evaluate()

    def __str__(self) -> str:
        return "(" + str(self.left) + " * " + str(self.right) + ")"


target = 64


def fitness_function(p: Expr) -> float:
    return abs(target - p.evaluate())


if __name__ == "__main__":
    g = extract_grammar([One, Add, Sub, Mul], Expr)
    alg = GP(
        g,
        treebased_representation,
        fitness_function,
        minimize=True,
        max_depth=8,
        number_of_generations=50,
        population_size=100,
        n_novelties=5,
    )
    (b, bf, bp) = alg.evolve(verbose=0)
    print(bf, bp, b)
