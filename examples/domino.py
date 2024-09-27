from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
from typing import Annotated, List
from typing import Any
import numpy as np
from sklearn.metrics import mean_squared_error
from geml.simplegp import SimpleGP
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.grammar import Grammar
from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.grammar.metahandlers.lists import ListSizeBetween
blacks = [(0, 6),
          (1, 0),
          (3, 5),
          (4, 1),
          (6, 0)]
top_target = [3,2,4,10,4,3,9]
side_target = [5,5,6,5,4,4,6]
def generateMatrix(x):
    result = []
    for i in range(0,x):
        temp = []
        for j in range(0,x):
            temp.append(0)
        result.append(temp)
    return result
class Domino(ABC):
    pass
class Board(ABC):
    pass
@dataclass
class DominoVertical(Domino):
    value = 1
    posX : Annotated[int, IntRange(0, 6)]
    posY : Annotated[int, IntRange(0, 5)]
    def __str__(self):
        return "("+ str(self.posX) +" "+ str(self.posY) +")"
@dataclass   
class DominoHorizontal(Domino):
    value = 2
    posX : Annotated[int, IntRange(1, 6)]
    posY : Annotated[int, IntRange(0, 6)]
    def __str__(self):
        return "("+ str(self.posX) +" "+ str(self.posY) +")"
@dataclass   
class GeneratedBoard(Board):
    dominos : Annotated[list[Domino], ListSizeBetween(20, 30)]
    def __str__(self):
        return str(self.dominos)
    
def fitness_function(n:Board):
    r = 0
    temp_top_target = [0] * 7
    temp_side_target = [0] * 7
    visited = generateMatrix(7)
    for i in n.dominos:
        if isinstance(i, DominoVertical):
            if visited[i.posY][i.posX] :
                r -= 5
            else:
                visited[i.posY][i.posX] = 1
            if visited[i.posY+1][i.posX] :
                r -= 5
            else:
                visited[i.posY+1][i.posX] = 1
            if (i.posX,i.posY) in blacks or (i.posX,i.posY+1) in blacks:
                r-= 10000
        else :
            if visited[i.posY][i.posX] :
                r -= 5
            else:
                visited[i.posY][i.posX] = 1
            if visited[i.posY][i.posX-1] :
                r -= 5
            else:
                visited[i.posY][i.posX-1] = 1
            if (i.posX,i.posY) in blacks or (i.posX-1,i.posY) in blacks:
                r-= 10000
        temp_top_target[i.posX] += i.value
        temp_side_target[i.posY] += i.value
    for i in range(0,7):
        if not abs(temp_top_target[i] - top_target[i]) :
            r+=10
        if not abs(temp_side_target[i] - side_target[i]):
            r+=10
    for i in range(0,7):
        for j in range(0,7):
            if (j,i) not in blacks and visited[j][i] == 0:
                r-= 5
    return r
def toboard(board):
    visited = generateMatrix(7)
    for i in blacks:
        visited[i[1]][i[0]] = -1
    for i in board.dominos:
        visited[i.posX][i.posY] += i.value
    result='\n'
    for i in visited:
        for j in i:
            result += str(j)
        result += "\n"
    return result
class DominoMatchBenchmark:
    def get_grammar(self) -> Grammar:
        return extract_grammar([GeneratedBoard, DominoHorizontal, DominoVertical], Board)
    def main(self, **args):
        g = self.get_grammar()
        alg = SimpleGP(
            grammar=g,
            minimize=False,
            fitness_function=fitness_function,
            crossover_probability=0.75,
            mutation_probability=0.01,
            max_depth=15,
            max_evaluations=20000,
            population_size=70,
            selection_method=("tournament", 2),
            elitism=5,
            **args,
        )
        best = alg.search()
        print(
            f"Fitness of {best.get_fitness(alg.get_problem())} by genotype: {toboard(best.genotype)} with phenotype: {best.get_phenotype()}",
        )
if __name__ == "__main__":
    DominoMatchBenchmark().main(seed=0)