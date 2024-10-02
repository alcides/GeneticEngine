from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated
import numpy as np


from geml.simplegp import SimpleGP
from geneticengine.grammar.decorators import weight
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.grammar import Grammar
from geneticengine.grammar.metahandlers.ints import IntRange

blacks = [
    (0, 6),
    (1, 0),
    (3, 5),
    (4, 1),
    (6, 0),
]
top_target = [3,2,4,10,4,3,9]
side_target = [5,5,6,5,4,4,6]

class Board(ABC):
    pass

class Domino(ABC):
    value : int
    posX : int
    posY : int

    @abstractmethod
    def processBoard(self, board:np.ndarray):
        pass

    def __str__(self):
            return "("+ str(self.posX) +", "+ str(self.posY) +")"

@dataclass
class DominoVertical(Domino):
    value = 1
    posX : Annotated[int, IntRange(0, 6)]
    posY : Annotated[int, IntRange(0, 5)]

    def processBoard(self, board:np.ndarray):
        board[self.posY, self.posX] +=1
        board[self.posY+1, self.posX] +=1


@dataclass
class DominoHorizontal(Domino):
    value = 2
    posX : Annotated[int, IntRange(1, 6)]
    posY : Annotated[int, IntRange(0, 6)]

    def processBoard(self, board:np.ndarray):
        board[self.posY, self.posX] +=1
        board[self.posY, self.posX-1] +=1

@weight(0.99)
@dataclass
class Add(Board):
    current : Domino
    next : Board

    def __str__(self):
        return str(self.current) +", "+ str(self.next)

@weight(0.01)
@dataclass
class Stop(Board):
    current : Domino

    def __str__(self):
        return str(self.current)

def fitness_function(n:Board):
    r = 0.0
    temp_target = np.zeros((2, 7))
    board = np.zeros((7, 7))
    #penalty for overlap
    p_overlap = 5
    #penalty for overlap black squares
    p_black = 100000
    #penalty for miss the target number
    p_target = 1
    #penalty for not visited a empty squares
    p_not_visited = 1
    obj = n
    for blackX, blackY in blacks:
        board[blackY, blackX] = p_black
    while isinstance(obj, Add) or isinstance(obj, Stop):
        i = obj.current
        i.processBoard(board)
        temp_target[0, i.posX] += i.value
        temp_target[1, i.posY] += i.value
        if isinstance(obj, Stop):
            break
        obj = obj.next
    for a in range(7):
        r += p_target * (2.0-((abs(temp_target[0, a] - top_target[a]))/top_target[a]))
        r += p_target * (2.0-((abs(temp_target[0, a] - side_target[a]))/side_target[a]))
    for elem in board.flatten():
        if elem >= p_black:
            if elem % p_black:
                r += elem
        elif not elem:
            r += p_not_visited
        else:
            r += (elem - 1) * p_overlap
    return r

def toboard(board):
    visited = np.zeros((7,7), dtype = int)
    for blackX, blackY in blacks:
        visited[blackY, blackX] = -10
    obj = board
    while True:
        i=obj.current
        visited[i.posY, i.posX] += i.value
        if isinstance(obj, Stop):
            break
        obj = obj.next
    result='\n'
    for i in visited:
        for j in i:
            result += str(j)+" "
        result += "\n"
    return result

class DominoMatchBenchmarkRecursive:

    def get_grammar(self) -> Grammar:
        return extract_grammar([Add, Stop, DominoHorizontal, DominoVertical], Board)

    def main(self, **args):
        g = self.get_grammar()

        alg = SimpleGP(
            grammar=g,
            minimize=True,
            fitness_function=fitness_function,
            crossover_probability=0.75,
            mutation_probability=0.01,
            max_depth=25,
            max_evaluations=10000,
            population_size=1000,
            selection_method=("tournament", 2),
            elitism=5,
            **args,
        )
        best = alg.search()
        print(
            f"Fitness of {best.get_fitness(alg.get_problem())} by genotype: {toboard(best.genotype)} with phenotype: {toboard(best.get_phenotype())}",
        )

if __name__ == "__main__":
    DominoMatchBenchmarkRecursive().main(seed=0)
