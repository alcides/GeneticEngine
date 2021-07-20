from dataclasses import dataclass
from enum import Enum
from typing import Annotated, List, Protocol, Tuple
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.treebased import treebased_representation
from geneticengine.core.tree import Node
from geneticengine.metahandlers.lists import ListSizeBetween
from geneticengine.algorithms.gp import GP


class Expr(Protocol):
    pass


class One(Node, Expr):
    def evaluate(self, **kwargs):
        return 1


@dataclass
class Add(Node, Expr):
    left: Node
    right: Node

    def evaluate(self, **kwargs):
        return self.left.evaluate() + self.right.evaluate()


if __name__ == "__main__":
    g = extract_grammar([One, Add], Expr)
    target = 5
    alg = GP(
        g,
        treebased_representation,
        lambda p: -abs(target - p.evaluate()),
        minimize=False,
        max_depth=10,
        number_of_generations=50,
        population_size=1500,
        novelty=50,
    )
    (b, bf) = alg.evolve()
    print(bf, b)
