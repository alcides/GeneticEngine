from dataclasses import dataclass
from enum import Enum
from typing import List, Protocol, Tuple
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.treebased import treebased_representation
from geneticengine.core.tree import Node
from geneticengine.metahandlers.lists import ListSizeBetween
from geneticengine.algorithms.gp.gp import GP


class Expr(Protocol):
    pass


class One(Node, Expr):
    def evaluate(self, **kwargs):
        return 1

    def size(self):
        return 1

    def __str__(self) -> str:
        return "1"


@dataclass
class Add(Node, Expr):
    left: Node
    right: Node

    def __init__(self,left,right):
        self.left = left
        self.right = right
    
    def evaluate(self, **kwargs):
        return self.left.evaluate() + self.right.evaluate()

    def size(self):
        return self.left.size() + self.right.size() + 1

    def __str__(self) -> str:
        return "(" + str(self.left) + " + " + str(self.right) + ")"

@dataclass
class Sub(Node, Expr):
    left: Node
    right: Node

    def __init__(self,left,right):
        self.left = left
        self.right = right
    
    def evaluate(self, **kwargs):
        return self.left.evaluate() - self.right.evaluate()

    def size(self):
        return self.left.size() + self.right.size() + 1

    def __str__(self) -> str:
        return "(" + str(self.left) + " - " + str(self.right) + ")"

@dataclass
class Mul(Node, Expr):
    left: Node
    right: Node

    def __init__(self,left,right):
        self.left = left
        self.right = right
    
    def evaluate(self, **kwargs):
        return self.left.evaluate() * self.right.evaluate()

    def size(self):
        return self.left.size() + self.right.size() + 1

    def __str__(self) -> str:
        return "(" + str(self.left) + " * " + str(self.right) + ")"


if __name__ == "__main__":
    g = extract_grammar([One, Add, Sub, Mul], Expr)
    target = 64
    alg = GP(
        g,
        treebased_representation,
        lambda p: (abs(target - p.evaluate()) - 1/p.size()),
        minimize=True,
        max_depth=8,
        number_of_generations=50,
        population_size=100,
        n_novelties=5,
    )
    (b, bf) = alg.evolve()
    print(bf, b)
