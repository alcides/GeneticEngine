from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Annotated


from geml.simplegp import SimpleGP
from geneticengine.grammar.decorators import weight
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

def generateMatrix(x):
    result = []
    for _ in range(0,x):
        temp = []
        for _ in range(0,x):
            temp.append(0)
        result.append(temp)
    return result

class Domino(ABC):
    value : int
    posX : int
    posY : int

class Board(ABC):
    pass

class Cal(ABC):
    pass

class Value(ABC):
    pass


#------------------------------------------------------------
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

    dominos : Annotated[list[Domino], ListSizeBetween(22, 23)]

    def __str__(self):
        return str(self.dominos)

@dataclass
class GeneratedBoardVertical(Board):

    dominos : Annotated[list[DominoVertical], ListSizeBetween(5,25)]

    def __str__(self):
        return str(self.dominos)

@dataclass
class GeneratedBoardHorizontal(Board):

    dominos : Annotated[list[DominoHorizontal], ListSizeBetween(5, 25)]

    def __str__(self):
        return str(self.dominos)
#------------------------------------------------------------------------------
@weight(0.99)
@dataclass
class Add(Board):
    current : Domino
    next : Board

    def __str__(self):
        return "("+ str(self.current)+")"

@weight(0.01)
@dataclass
class Stop(Board):
    current : Domino

    def __str__(self):
        return "("+ str(self.current)+")"

def fitness_function(n:GeneratedBoard):
    r = 0
    temp_top_target = [0] * 7
    temp_side_target = [0] * 7
    #penalty for overlap
    p_overlap = 1
    #penalty for overlap black squares
    p_black = 10000
    p_sameplace = 10000
    #penalty for miss the target number
    p_target = 1
    #penalty for not visited a empty squares
    p_not_visited = 1
    #gains for hit the target number
    g_target = 0
    #gains for visit a new empty square
    g_new = 0
    visited = generateMatrix(7)
    squares = []
    for i in n.dominos:
        if visited[i.posY][i.posX] :
            r -= p_overlap
        else:
            visited[i.posY][i.posX] = 1
            r += g_new
        if (i.posX,i.posY) in blacks:
                r-= p_black
        if isinstance(i, DominoVertical):
            if visited[i.posY+1][i.posX] :
                r -= p_overlap
            else:
                visited[i.posY+1][i.posX] = 1
                r += g_new
            if (i.posX,i.posY+1) in blacks:
                r-= p_black
        else :
            if visited[i.posY][i.posX-1] :
                r -= p_overlap
            else:
                visited[i.posY][i.posX-1] = 1
                r += g_new
            if (i.posX-1,i.posY) in blacks:
                r-= p_black
        if (i.posX, i.posY) in squares:
            r -= p_sameplace
        else:
            squares.append((i.posX,i.posY))
        temp_top_target[i.posX] += i.value
        temp_side_target[i.posY] += i.value
    for a in range(0,7):
        top =abs(temp_top_target[a] - top_target[a])
        if top :
            r-=p_target*top
        else:
            r+=g_target
        side = abs(temp_side_target[a] - side_target[a])
        if side:
            r-=p_target*side
        else:
            r+=g_target
    for b in range(0,7):
        for j in range(0,7):
            if (j,b) not in blacks and visited[j][b] == 0:
                r-= p_not_visited
    return r

def fitness_function_v3(n:Board):
    r = 0.0
    temp_top_target = [0] * 7
    temp_side_target = [0] * 7
    #penalty for overlap
    p_overlap = 5
    #penalty for overlap black squares
    p_black = 100000
    p_sameplace = 100000
    #penalty for miss the target number
    p_target = 1
    #penalty for not visited a empty squares
    p_not_visited = 1
    #gains for hit the target number
    g_target = 0
    #gains for visit a new empty square
    g_new = 0
    visited = generateMatrix(7)
    squares = []
    obj = n
    while isinstance(obj, Add) or isinstance(obj, Stop):
        i = obj.current
        if visited[i.posY][i.posX] :
            r -= p_overlap
        else:
            visited[i.posY][i.posX] = 1
            r += g_new
        if (i.posX,i.posY) in blacks:
                r-= p_black
        if isinstance(i, DominoVertical):
            if visited[i.posY+1][i.posX] :
                r -= p_overlap
            else:
                visited[i.posY+1][i.posX] = 1
                r += g_new
            if (i.posX,i.posY+1) in blacks:
                r-= p_black
        else :
            if visited[i.posY][i.posX-1] :
                r -= p_overlap
            else:
                visited[i.posY][i.posX-1] = 1
                r += g_new
            if (i.posX-1,i.posY) in blacks:
                r-= p_black
        if (i.posX, i.posY) in squares:
            r -= p_sameplace
        else:
            squares.append((i.posX,i.posY))
        temp_top_target[i.posX] += i.value
        temp_side_target[i.posY] += i.value
        if isinstance(obj, Stop):
            break
        obj = obj.next
    for a in range(0,7):
        top =abs(temp_top_target[a] - top_target[a])
        if top :
            r-=p_target * ((top+1.0)/top_target[a])
        else:
            r+=g_target
        side = abs(temp_side_target[a] - side_target[a])
        if side:
            r-=p_target* ((side+1.0)/side_target[a])
        else:
            r+=g_target
    for b in range(0,7):
        for j in range(0,7):
            if (j,b) not in blacks and visited[j][b] == 0:
                r-= p_not_visited
    r+=2*len(squares)
    return r


def toboard(board):
    visited = generateMatrix(7)
    for i in blacks:
        visited[i[1]][i[0]] = -10
    for i in board.dominos:
        visited[i.posY][i.posX] += i.value
    result='\n'
    for i in visited:
        for j in i:
            result += str(j)+" "
        result += "\n"
    return result

def toboardv3(board):
    visited = generateMatrix(7)
    for i in blacks:
        visited[i[1]][i[0]] = "X"
    obj = board
    while True:
        i=obj.current
        visited[i.posY][i.posX] += i.value
        if isinstance(obj, Stop):
            break
        obj = obj.next
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
            minimize=False,
            fitness_function=fitness_function,
            crossover_probability=0.75,
            mutation_probability=0.01,
            max_evaluations=500000000000,
            max_depth=20,
            max_time=60,
            csv_output= "Hello.csv",
            population_size=50,
            selection_method=("tournament", 2),
            elitism=5,
            **args,
        )
        best = alg.search()
        print(
            f"Fitness of {best.get_fitness(alg.get_problem())} by genotype: {toboard(best.genotype)} with phenotype: {toboard(best.get_phenotype())}",
        )

class DominoMatchBenchmarkV3:

    def get_grammar(self) -> Grammar:
        return extract_grammar([Add, Stop, DominoHorizontal, DominoVertical], Board)

    def main(self, **args):
        g = self.get_grammar()

        alg = SimpleGP(
            grammar=g,
            minimize=False,
            fitness_function=fitness_function_v3,
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
            f"Fitness of {best.get_fitness(alg.get_problem())} by genotype: {toboardv3(best.genotype)} with phenotype: {toboardv3(best.get_phenotype())}",
        )


if __name__ == "__main__":
    DominoMatchBenchmarkV3().main(seed=0)
