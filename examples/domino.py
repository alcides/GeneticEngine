from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated
import numpy as np


from geml.simplegp import SimpleGP
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.grammar import Grammar
from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.grammar.metahandlers.lists import ListSizeBetween

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
            return "("+ str(self.posX) +" "+ str(self.posY) +")"


#------------------------------------------------------------
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



@dataclass
class GeneratedBoard(Board):

    dominos : Annotated[list[Domino], ListSizeBetween(22, 23)]

    def __str__(self):
        return str(self.dominos)

#------------------------------------------------------------------------------

def fitness_function(n:GeneratedBoard):
    temp_target = np.zeros((2, 7))
    board = np.zeros((7, 7))
    #penalty for overlap
    p_overlap = 1
    #penalty for overlap black squares
    p_black = 10000
    #penalty for miss the target number
    p_target = 1
    #penalty for not visited a empty squares
    p_not_visited = 1
    for blackX, blackY in blacks:
        board[blackY, blackX] = p_black
    for i in n.dominos:
        i.processBoard(board)
        temp_target[0, i.posX] += i.value
        temp_target[1, i.posY] += i.value
    r_p_top_target = [p_target * ((2.0-((abs(temp_target[0, a] - top_target[a]))/top_target[a]))%2) for a in range(7)]
    r_p_side_target = [p_target * ((2.0-((abs(temp_target[1, a] - side_target[a]))/side_target[a]))%2) for a in range(7)]
    r_p_board =[
        elem if elem > p_black
        else 0 if elem == p_black
        else p_not_visited if elem == 0
        else (elem - 1) * p_overlap
        for elem in board.flatten()
    ]
    return sum(r_p_top_target) + sum(r_p_side_target) + sum(r_p_board)


def toboard(board):
    visited = np.zeros((7,7), dtype = int)
    for i in blacks:
        visited[i[1], i[0]] = -10
    for i in board.dominos:
        visited[i.posY, i.posX] += i.value
    result='\n'
    for i in visited:
        for j in i:
            result += str(j)+" "
        result += "\n"
    return result

class DominoMatchBenchmark:

    def get_grammar(self) -> Grammar:
        return extract_grammar([GeneratedBoard, DominoHorizontal, DominoVertical], Board)

    def main(self, **args):
        g = self.get_grammar()

        alg = SimpleGP(
            grammar=g,
            minimize=True,
            fitness_function=fitness_function,
            crossover_probability=0.75,
            mutation_probability=0.01,
            max_evaluations=10000,
            max_depth=20,
            max_time=60,
            population_size=50,
            selection_method=("tournament", 2),
            elitism=5,
            **args,
        )
        best = alg.search()
        print(
            f"Fitness of {best.get_fitness(alg.get_problem())} by genotype: {toboard(best.genotype)} with phenotype: {toboard(best.get_phenotype())}",
        )

if __name__ == "__main__":
    DominoMatchBenchmark().main(seed=0)
