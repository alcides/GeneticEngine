from abc import ABC
from dataclasses import dataclass
from textwrap import indent
from typing import Annotated, List, Callable, Any
from geneticengine.grammars.coding.classes import Condition, Statement
from geneticengine.metahandlers.ints import IntRange



@dataclass
class Code(Statement):
    stmts: List[Statement]

    def evaluate(self, x: float = 1) -> float:
        for stmt in self.stmts:
            x = stmt.evaluate(x)
        return x

    def evaluate_lines(self, **kwargs) -> Callable[[Any], float]:
        def ev(line): 
            for stmt in self.stmts:
                x = stmt.evaluate_lines(**kwargs)(line)
            return x
        return lambda line: ev(line)

    def __str__(self):
        return "\n".join([str(stmt) for stmt in self.stmts])


@dataclass
class ForLoop(Statement):
    iterationRange: Annotated[int, IntRange(1, 6)]
    loopedCode: Statement

    def evaluate(self, x: float = 1) -> float:
        for _ in range(self.iterationRange):
            x = self.loopedCode.evaluate(x)
        return x
    
    def evaluate_lines(self, **kwargs) -> Callable[[Any], float]:
        def ev(line): 
            for _ in range(self.iterationRange):
                x = self.loopedCode.evaluate_lines(**kwargs)(line)
            return x
        return lambda line: ev(line)

    def __str__(self):
        return "for i in range({}):\n{}".format(
            self.iterationRange, indent(str(self.loopedCode), "\t")
        )


@dataclass
class While(Statement):
    cond: Condition
    loopedCode: Statement

    def evaluate(self, x: float = 1) -> float:
        for _ in range(self.cond.evaluate(x)):
            x = self.loopedCode.evaluate(x)
        return x
    
    def evaluate_lines(self, **kwargs) -> Callable[[Any], float]:
        def ev(line): 
            while (self.cond.evaluate_lines(**kwargs)(line)):
                x = self.loopedCode.evaluate_lines(**kwargs)(line)
            return x
        return lambda line: ev(line)

    def __str__(self):
        return "while {}:\n{}".format(
            self.cond, indent(str(self.loopedCode), "\t")
        )



@dataclass
class IfThen(Statement):
    cond: Condition
    then: Statement

    def evaluate(self, x: float = 1) -> float:
        if self.cond.evaluate(x):
            x = self.then.evaluate(x)
        return x

    def evaluate_lines(self, **kwargs) -> Callable[[Any], float]:
        def ev(line): 
            if self.cond.evaluate_lines(**kwargs)(line):
                x = self.then.evaluate_lines(**kwargs)(line)
            else:
                x = 0 # Is this correctly done?
            return x
        return lambda line: ev(line)

    def __str__(self):
        return "if {}:\n{}".format(
            self.cond, indent(str(self.then), "\t")
        )

@dataclass
class IfThenElse(Statement):
    cond: Condition
    then: Statement
    elze: Statement

    def evaluate(self, x: float = 1) -> float:
        if self.cond.evaluate(x):
            x = self.then.evaluate(x)
        else:
            x = self.elze.evaluate(x)
        return x

    def evaluate_lines(self, **kwargs) -> Callable[[Any], float]:
        def ev(line): 
            if self.cond.evaluate_lines(**kwargs)(line):
                z = self.then.evaluate_lines(**kwargs)(line)
            else:
                z = self.elze.evaluate_lines(**kwargs)(line)
            return z
        return lambda line: ev(line)

    def __str__(self):
        return "if {}:\n{}\nelse:\n{}".format(
            self.cond, indent(str(self.then), "\t"), indent(str(self.elze), "\t")
        )


