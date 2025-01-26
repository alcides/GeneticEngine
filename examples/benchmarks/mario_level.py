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
from geneticengine.grammar.metahandlers.vars import VarRange
from geneticengine.problems import MultiObjectiveProblem, Problem


X = Annotated[int, IntRange(5, 95)]
Y = Annotated[int, IntRange(3, 5)]
W = Annotated[int, IntRange(3, 15)]
Wc = Annotated[int, IntRange(3, 15)]
Wg = Annotated[int, IntRange(2, 5)]
Wb = Annotated[int, IntRange(2, 7)]
H = Annotated[int, IntRange(2, 3)]


# Based on Evolving Levels for Super Mario Bros Using Grammatical Evolution


class Chunk(ABC):

    @abstractmethod
    def centrail_point(self) -> list[tuple[float, float]]: ...

    @abstractmethod
    def place_in(self, map: np.ndarray): ...


@dataclass
class Platform(Chunk):
    x: X
    y: Y
    w: W

    def centrail_point(self) -> list[tuple[float, float]]:
        c1 = (self.x + self.w / 2, self.y + 0.5)
        return [c1]

    def place_in(self, map: np.ndarray):
        # place the tube itself
        for i in range(0, self.w):
            map[self.x + i, self.y] += 1


@dataclass
class Hill(Chunk):
    x: X
    y: Y
    w: W

    def centrail_point(self) -> list[tuple[float, float]]:
        c1 = (self.x + self.w / 2, self.y + 0.5)
        return [c1]

    def place_in(self, map: np.ndarray):
        # place the tube itself
        for i in range(0, self.w):
            map[self.x + i, self.y] += 1


@dataclass
class Gap(Chunk):
    x: X
    y: Y
    wg: Wg
    wbefore: W
    wafter: W

    def centrail_point(self) -> list[tuple[float, float]]:
        c1 = (self.x - self.wbefore / 2, self.y + 0.5)
        c2 = (self.x - self.wbefore + self.wg + self.wafter / 2, self.y + 0.5)
        return [c1, c2]

    def place_in(self, map: np.ndarray):

        # place platform before
        for i in range(0, self.wbefore):
            map[self.x - i, self.y] += 1

        # place platform after
        for i in range(0, self.wafter):
            map[self.x + i, self.y] += 1


@dataclass
class CannonHill(Chunk):
    x: X
    y: Y
    h: H
    wbefore: W
    wafter: W

    def centrail_point(self) -> list[tuple[float, float]]:
        c1 = (self.x - self.wbefore / 2, self.y + self.h / 2)
        c2 = (self.x - self.wbefore + self.wafter / 2, self.y + self.h / 2)
        return [c1, c2]

    def place_in(self, map: np.ndarray):
        # place the tube itself
        for i in range(0, self.h):
            map[self.x, self.y + i] += 1

        # place platform before
        for i in range(0, self.wbefore):
            map[self.x - i, self.y] += 1

        # place platform after
        for i in range(0, self.wafter):
            map[self.x + i, self.y] += 1


@dataclass
class TubeHill(Chunk):
    x: X
    y: Y
    h: H
    wbefore: W
    wafter: W

    def centrail_point(self) -> list[tuple[float, float]]:
        c1 = (self.x - self.wbefore / 2, self.y + self.h / 2)
        c2 = (self.x - self.wbefore + self.wafter / 2, self.y + self.h / 2)
        return [c1, c2]

    def place_in(self, map: np.ndarray):
        # place the tube itself
        for i in range(0, self.h):
            map[self.x, self.y + i] += 1

        # place platform before
        for i in range(0, self.wbefore):
            map[self.x - i, self.y] += 1

        # place platform after
        for i in range(0, self.wafter):
            map[self.x + i, self.y] += 1


@dataclass
class Coin(Chunk):
    x: X
    y: Y
    w: W

    def centrail_point(self) -> list[tuple[float, float]]:
        c1 = (self.x + self.w / 2, self.y + 0.5)
        return [c1]

    def place_in(self, map: np.ndarray):
        # place the tube itself
        for i in range(0, self.w):
            map[self.x + i, self.y] += 1


@dataclass
class Cannon(Chunk):
    x: X
    y: Y
    h: H
    wbefore: W
    wafter: W

    def centrail_point(self) -> list[tuple[float, float]]:
        c1 = (self.x - self.wbefore / 2, self.y + self.h / 2)
        c2 = (self.x - self.wbefore + self.wafter / 2, self.y + self.h / 2)
        return [c1, c2]

    def place_in(self, map: np.ndarray):
        # place the tube itself
        for i in range(0, self.h):
            map[self.x, self.y + i] += 1

        # place platform before
        for i in range(0, self.wbefore):
            map[self.x - i, self.y] += 1

        # place platform after
        for i in range(0, self.wafter):
            map[self.x + i, self.y] += 1


@dataclass
class Tube(Chunk):
    x: X
    y: Y
    h: H
    wbefore: W
    wafter: W

    def centrail_point(self) -> list[tuple[float, float]]:
        c1 = (self.x - self.wbefore / 2, self.y + self.h / 2)
        c2 = (self.x - self.wbefore + self.wafter / 2, self.y + self.h / 2)
        return [c1, c2]

    def place_in(self, map: np.ndarray):
        # place the tube itself
        for i in range(0, self.h):
            map[self.x, self.y + i] += 1

        # place platform before
        for i in range(0, self.wbefore):
            map[self.x - i, self.y] += 1

        # place platform after
        for i in range(0, self.wafter):
            map[self.x + i, self.y] += 1


@dataclass
class Box:
    kind: Annotated[str, VarRange(["bcoin", "bpowerup", "rcoin", "rock"])]
    x: X
    y: Y

    def centrail_point(self) -> list[tuple[float, float]]:
        c1 = (self.x, self.y + 0.5)
        return [c1]

    def place_in(self, map: np.ndarray):
        map[self.x, self.y] += 1


@dataclass
class Boxes(Chunk):
    boxes: Annotated[list[Box], ListSizeBetween(2, 6)]

    def centrail_point(self) -> list[tuple[float, float]]:
        return list(*[box.centrail_point() for box in self.boxes])

    def place_in(self, map: np.ndarray):
        for box in self.boxes:
            box.place_in(map)


@dataclass
class Enemy:
    kind: Annotated[str, VarRange(["goomba", "koopa"])]
    x: X

    def centrail_point(self) -> list[tuple[float, float]]:
        c1 = (self.x + 0.5, 0.5)
        return [c1]

    def place_in(self, map: np.ndarray):
        map[self.x, 0] = 1


map_w = 110
map_h = 10


@dataclass
class Level:
    chunks: list[Chunk]
    enemies: Annotated[list[Enemy], ListSizeBetween(2, 10)]

    def number_of_chunks(self) -> int:
        max_chunks = map_w * map_h
        return max_chunks - len(self.chunks)

    def conflicts(self) -> int:
        map = np.zeros((map_w, map_h))

        for chunk in self.chunks:
            chunk.place_in(map)
        for enemy in self.enemies:
            enemy.place_in(map)

        return np.sum(map)


class MarioBenchmark(Benchmark):
    def __init__(self):
        self.setup_problem()
        self.setup_grammar()

    def setup_problem(self):

        # Problem
        def fitness_function(b: Level) -> list[int]:
            return [b.number_of_chunks(), b.conflicts()]

        self.problem = MultiObjectiveProblem(minimize=[True, True], fitness_function=fitness_function, target=[0, 0])

    def setup_grammar(self):
        self.grammar = extract_grammar(
            [Level, Chunk, Enemy, TubeHill, Box, Coin, Gap, Platform, Hill, CannonHill, Cannon, Tube, Boxes, Box],
            Level,
        )

    def get_problem(self) -> Problem:
        return self.problem

    def get_grammar(self) -> Grammar:
        return self.grammar


if __name__ == "__main__":
    example_run(MarioBenchmark())
