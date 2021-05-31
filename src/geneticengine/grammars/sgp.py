from geneticengine.core.tree import Node

from typing import Protocol


class Number(Protocol):
    pass


class Plus(Node, Number):
    def __init__(self, left: Number, right: Number):
        self.left = left
        self.right = right

    def evaluate(self, *args):
        return self.left.evaluate(*args) + self.right.evaluate(*args)


class Literal(Node, Number):
    def __init__(self, val: int):
        self.val = val

    def evaluate(self, *args):
        return self.val
