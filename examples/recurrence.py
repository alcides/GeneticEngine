from abc import ABC
from dataclasses import dataclass
from typing import Annotated, List
import numpy as np
from scipy import rand
from geneticengine.algorithms.gp.callback import DebugCallback
from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.vars import VarRange

class Node(ABC):
    pass

    def evaluate(self, input:list[int]):
        ...

@dataclass
class Op(Node):
    r:Node
    op:Annotated[str, VarRange(["+", "-", "*", "/"])]
    l:Node

    def evaluate(self, input:list[int]):
        if self.op == "+":
            return self.r.evaluate(input) + self.l.evaluate(input)
        elif self.op == "-":
            return self.r.evaluate(input) - self.l.evaluate(input)
        elif self.op == "*":
            return self.r.evaluate(input) - self.l.evaluate(input)
        else:
            if self.l.evaluate(input) == 0:
                return 0
            return self.r.evaluate(input) / self.l.evaluate(input)

@dataclass
class Access(Node):
    i:Annotated[int, IntRange(-2,-1)]

    def evaluate(self, input:list[int]):
        return input[self.i]


@dataclass
class Literal(Node):
    i:Annotated[int, IntRange(-10,10)]

    def evaluate(self, input:list[int]):
        return self.i


def fitness_function(p):
    dataset = [1,1,2,3,5,8,13]

    errors = 0
    for i in range(2, len(dataset)):
        input = dataset[:i]
        r = p.evaluate(input)
        e = abs(r - dataset[i])
        errors += e**2
    return errors


if __name__ == "__main__":
    g = extract_grammar([Op, Access, Literal], Node)
    gp = GP(grammar=g, evaluation_function=fitness_function, 
            minimize=True,
            max_depth=10, number_of_generations=100, population_size=10, probability_mutation=0.5, probability_crossover=0.4)
    (_, fitness, explanation) = gp.evolve()
    print(fitness, explanation)


