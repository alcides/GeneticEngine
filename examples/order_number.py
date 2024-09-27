from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Annotated, List
from typing import Any

import numpy as np
from sklearn.metrics import mean_squared_error

from geml.simplegp import SimpleGP
from geneticengine.grammar.decorators import weight
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.grammar import Grammar
from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.grammar.metahandlers.lists import ListSizeBetween

er = 0
count = 0

class N(ABC):
    pass

@dataclass
class A(N):
    value: Annotated[int, IntRange(0,100)]
    value2: Annotated[int, IntRange(0,100)]
    value3: Annotated[int, IntRange(0,100)]

    def __str__(self):
        return str(self.value) + " > " + str(self.value2) + " > " + str(self.value3) 
    
@dataclass
class B(N):
    value: Annotated[int, IntRange(0,1000)]
    value2: Annotated[int, IntRange(0,1000)]
    value3: Annotated[int, IntRange(0,1000)]
    value4: Annotated[int, IntRange(0,1000)]
    value5: Annotated[int, IntRange(0,1000)]

    def __str__(self):
        return str(self.value) + " > " + str(self.value2) + " > " + str(self.value3) + " > " + str(self.value4) + " > " + str(self.value5)
def fitness_3(n):
    r = 0
    if n.value - n.value2 <= 0:
        r = 1
    if n.value2 - n.value3 <= 0:
        r += 2
    return r

def fitness_5(n):
    r = 0
    if n.value - n.value2 <= 0:
        r += 8
    if n.value2 - n.value3 <= 0:
        r += 4
    if n.value3 - n.value4 <= 0:
        r += 2
    if n.value4 - n.value5 <= 0:
        r += 1
    return r


class NumberMatchBenchmark:

    def get_grammar(self) -> Grammar:
        return extract_grammar([A], N)

    def main(self, **args):
        g = self.get_grammar()
        global count
        alg = SimpleGP(
            grammar=g,
            minimize=True,
            fitness_function=fitness_3,
            crossover_probability=0.75,
            mutation_probability=0.01,
            max_depth=15,
            max_evaluations=10,
            max_time= 600,
            population_size=50,
            csv_output= "output/order"+ str(count) +".csv",
            selection_method=("tournament", 2),
            elitism=5,
            **args,
        )
        best = alg.search()
        be = best.get_fitness(alg.get_problem())
        print(
            f"Fitness of {be} by genotype: {best.genotype} with phenotype: {best.get_phenotype()}",
        )



if __name__ == "__main__":
    for i in range(30):
        NumberMatchBenchmark().main(seed=i)
        count+=1
    print(er)
    