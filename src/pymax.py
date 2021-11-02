from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from textwrap import indent
from typing import Annotated, List, Protocol, Union
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.treebased import treebased_representation
from geneticengine.core.tree import Node
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.lists import ListSizeBetween
from geneticengine.algorithms.gp.gp import GP
from number_maker import Mul

class Expr(Protocol):
    pass

@dataclass
class ForLoop(Node,Expr):
    iterationRange: Annotated[int, IntRange(1, 6)]
    loopedCode: Node

    def __init__(self,iterationRange,loopedCode):
        self.iterationRange = iterationRange
        self.loopedCode = loopedCode
    
    def __str__(self):
        return 'for i in range({}):\n{}'.format(self.iterationRange,indent(str(self.loopedCode),'\t'))


            

class OneHalve(Node,Expr):
    def evaluate(self, **kwargs):
        return 0.5

    def __str__(self) -> str:
        return "0.5"

@dataclass
class PlusOneHalve(Node,Expr):
    def __init__(self):
        pass

    def __str__(self) -> str:
        return "x = (" + 'x' + " + " + str(OneHalve()) + ")"

class MultOneHalve(Node,Expr):
    
    def __str__(self) -> str:
        return "x = (" + 'x' + " * " + str(OneHalve()) + ")"



def fit(indiv):
    code = 'x = 0\n' + str(indiv)
    loc = {}
    exec(code, globals(), loc)
    x = loc['x']
    return x

fitness_function = lambda x: fit(x)

if __name__ == "__main__":
    g = extract_grammar([PlusOneHalve,MultOneHalve,ForLoop],Expr)
    alg = GP(
        g,
        treebased_representation,
        fitness_function,
        max_depth=5,
        max_init_depth=5,
        population_size=40,
        number_of_generations=3,
        minimize=False
    )
    (b,bf) = alg.evolve(verbose=1)
    print(b)
    print("With fitness: {}".format(bf))
