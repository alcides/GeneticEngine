from abc import ABC
from dataclasses import dataclass
from textwrap import indent
from typing import Annotated, List
from geneticengine.grammars.coding.classes import Condition, Statement
from geneticengine.metahandlers.ints import IntRange



@dataclass
class Code(Statement):
    stmts: List[Statement]

    def evaluate(self, x: float = 1.0) -> float:
        for stmt in self.stmts:
            x = stmt.evaluate(x)
        return x

    def __str__(self):
        return "\n".join([str(stmt) for stmt in self.stmts])


@dataclass
class ForLoop(Statement):
    iterationRange: Annotated[int, IntRange(1, 6)]
    loopedCode: Statement

    def evaluate(self, x: float = 1.0) -> float:
        for _ in range(self.iterationRange):
            x = self.loopedCode.evaluate(x)
        return x

    def __str__(self):
        return "for i in range({}):\n{}".format(
            self.iterationRange, indent(str(self.loopedCode), "\t")
        )


@dataclass
class While(Statement):
    cond: Condition
    loopedCode: Statement

    def evaluate(self, x: bool = False,y: float = 1.0) -> float:
        while self.cond.evaluate(x):
            z = self.loopedCode.evaluate(y)
        return z

    def __str__(self):
        return "while ({}):\n{}".format(
            self.cond, indent(str(self.loopedCode), "\t")
        )

@dataclass
class IfThen(Statement):
    cond: Condition
    then: Statement

    def evaluate(self, x: bool = False,y: float = 1.0) -> float:
        if self.cond.evaluate(x):
            z = self.then.evaluate(y)
        return z

    def __str__(self):
        return "if ({}):\n{}".format(
            self.cond, indent(str(self.then), "\t")
        )

@dataclass
class IfThenElse(Statement):
    cond: Condition
    then: Statement
    elze: Statement

    def evaluate(self, x: bool = False,y: float = 1.0) -> float:
        if self.cond.evaluate(x):
            z = self.then.evaluate(y)
        else:
            z = self.elze.evaluate(y)
        return z

    def __str__(self):
        return "if ({}):\n{}\nelse:\n{}".format(
            self.cond, indent(str(self.then), "\t"), indent(str(self.elze), "\t")
        )


