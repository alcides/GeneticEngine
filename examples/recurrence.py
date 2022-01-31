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

class Node(ABC):
    pass

    def evaluate(self, input:List[int]):
        ...

@dataclass
class Plus(Node):
    r:Node
    l:Node

    def evaluate(self, input:List[int]):
        return self.r.evaluate(input) + self.l.evaluate(input)

@dataclass
class Access(Node):
    i:Annotated[int, IntRange(-2,-1)]

    def evaluate(self, input:List[int]):
        return input[self.i]


@dataclass
class Literal(Node):
    i:Annotated[int, IntRange(-10,10)]

    def evaluate(self, input:List[int]):
        return self.i


def fitness_function(p):
    dataset = [1,1,2,3,5,8,13]

    errors = []
    for i in range(2, len(dataset)):
        input = dataset[:i]
        r = p.evaluate(input)
        e = abs(r - dataset[i])
        errors.append(e ** 2)
    return np.mean(errors)


if __name__ == "__main__":
    

    g = extract_grammar([Plus, Access], Node)
    r = RandomSource(seed=123)
    gp = GP(grammar=g, evaluation_function=fitness_function, randomSource=lambda x: r, 
            max_depth=4, number_of_generations=100, population_size=100, probability_mutation=1, probability_crossover=1,
            callbacks=[DebugCallback()])
    (_, fitness, explanation) = gp.evolve()
    print(fitness, explanation)


