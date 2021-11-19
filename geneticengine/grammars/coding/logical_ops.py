from abc import ABC
from dataclasses import dataclass
from typing import Annotated, List
from geneticengine.metahandlers.ints import IntRange
from geneticengine.grammars.coding.classes import Condition



@dataclass
class And(Condition):
    left: Condition
    right: Condition
    
    def evaluate(self, x: bool = False) -> bool:
        return self.left.evaluate(x) and self.right.evaluate(x)

    def __str__(self):
        return f"({self.left} and {self.right})"

@dataclass
class Or(Condition):
    left: Condition
    right: Condition
    
    def evaluate(self, x: bool = False) -> bool:
        return self.left.evaluate(x) or self.right.evaluate(x)

    def __str__(self):
        return f"({self.left} or {self.right})"

@dataclass
class Not(Condition):
    cond: Condition
    
    def evaluate(self, x: bool = False) -> bool:
        return not self.cond(x)

    def __str__(self):
        return f"(not {self.cond})"
