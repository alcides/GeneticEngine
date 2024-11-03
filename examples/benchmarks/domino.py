from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated

import numpy as np


from examples.benchmarks.benchmark import Benchmark, example_run
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.grammar import Grammar
from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.grammar.metahandlers.lists import ListSizeBetween
from geneticengine.problems import Problem
from geneticengine.problems import SingleObjectiveProblem

blacks = [
    (0, 6),
    (1, 0),
    (3, 5),
    (4, 1),
    (6, 0),
]
top_target = [3, 2, 4, 10, 4, 3, 9]
side_target = [5, 5, 6, 5, 4, 4, 6]


class Domino(ABC):
    value: int
    posX: int
    posY: int

    @abstractmethod
    def processBoard(self, board: np.ndarray):
        pass

    def __str__(self):
        return "(" + str(self.posX) + " " + str(self.posY) + ")"


@dataclass
class DominoVertical(Domino):
    value = 1
    posX: Annotated[int, IntRange(0, 6)]
    posY: Annotated[int, IntRange(0, 5)]

    def processBoard(self, board: np.ndarray):
        board[self.posY, self.posX] += 1
        board[self.posY + 1, self.posX] += 1


@dataclass
class DominoHorizontal(Domino):
    value = 2
    posX: Annotated[int, IntRange(1, 6)]
    posY: Annotated[int, IntRange(0, 6)]

    def processBoard(self, board: np.ndarray):
        board[self.posY, self.posX] += 1
        board[self.posY, self.posX - 1] += 1


@dataclass
class Board:

    dominos: Annotated[list[Domino], ListSizeBetween(22, 22)]

    def __str__(self):
        visited = np.zeros((7, 7), dtype=int)
        for i in blacks:
            visited[i[1], i[0]] = -10
        for i in self.dominos:
            visited[i.posY, i.posX] += i.value
        result = "\n"
        for i in visited:
            for j in i:
                result += str(j) + " "
            result += "\n"
        return result


class DominoBenchmark(Benchmark):
    def __init__(self, blacks, top_target, side_target):
        self.setup_problem(blacks, top_target, side_target)
        self.setup_grammar()

    def setup_problem(self, blacks, top_target, side_target):

        # Problem
        def fitness_function(b: Board):
            board = np.zeros((7, 7))
            temp_target = np.zeros((2, 7))
            # penalty for overlap
            p_overlap = 1
            # penalty for overlap black squares
            p_black = 10000
            # penalty for miss the target number
            p_target = 1
            # penalty for not visited a empty squares
            p_not_visited = 1
            for blackX, blackY in blacks:
                board[blackY, blackX] = p_black
            for i in b.dominos:
                i.processBoard(board)
                temp_target[0, i.posX] += i.value
                temp_target[1, i.posY] += i.value
            r_p_top_target = [
                p_target * ((2.0 - ((abs(temp_target[0, a] - top_target[a])) / top_target[a])) % 2) for a in range(7)
            ]
            r_p_side_target = [
                p_target * ((2.0 - ((abs(temp_target[1, a] - side_target[a])) / side_target[a])) % 2) for a in range(7)
            ]
            r_p_board = [
                (
                    elem
                    if elem > p_black
                    else 0 if elem == p_black else p_not_visited if elem == 0 else (elem - 1) * p_overlap
                )
                for elem in board.flatten()
            ]
            return sum(r_p_top_target) + sum(r_p_side_target) + sum(r_p_board)

        self.problem = SingleObjectiveProblem(minimize=True, fitness_function=fitness_function, target=0)

    def setup_grammar(self):
        self.grammar = extract_grammar([DominoVertical, DominoHorizontal, Domino, Board], Board)

    def get_problem(self) -> Problem:
        return self.problem

    def get_grammar(self) -> Grammar:
        return self.grammar


if __name__ == "__main__":
    example_run(DominoBenchmark(blacks, top_target, side_target))
