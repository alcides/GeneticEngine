from geneticengine.core.tree import Node
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.vars import VarRange


from typing import Annotated, Protocol


class Number(Protocol):
    pass


class Plus(Node, Number):
    def __init__(self, left: Number, right: Number):
        self.left = left
        self.right = right

    def evaluate(self, **kwargs):
        return self.left.evaluate(**kwargs) + self.right.evaluate(**kwargs)

    def __repr__(self):
        return f"({self.left} + {self.right})"


class Mul(Node, Number):
    def __init__(self, left: Number, right: Number):
        self.left = left
        self.right = right

    def evaluate(self, **kwargs):
        return self.left.evaluate(**kwargs) * self.right.evaluate(**kwargs)

    def __repr__(self):
        return f"({self.left} * {self.right})"


class SafeDiv(Node, Number):
    def __init__(self, left: Number, right: Number):
        self.left = left
        self.right = right

    def evaluate(self, **kwargs):
        d1 = self.left.evaluate(**kwargs)
        d2 = self.right.evaluate(**kwargs)
        if d2 == 0:
            d2 = 0.000001
        return d1 / d2

    def __repr__(self):
        return f"({self.left} / {self.right})"


class Literal(Node, Number):
    def __init__(self, val: Annotated[int, IntRange(-10, 11)]):
        self.val = val

    def evaluate(self, **kwargs):
        return self.val

    def __repr__(self):
        return str(self.val)


class Var(Node, Number):
    def __init__(self, name: Annotated[str, VarRange(["x", "y", "z"])]):
        self.name = name

    def evaluate(self, **kwargs):
        return kwargs[self.name]

    def __repr__(self):
        return self.name