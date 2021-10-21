from dataclasses import dataclass
from enum import Enum
from typing import Annotated, List, Protocol, Tuple
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.treebased import treebased_representation
from geneticengine.core.tree import Node
from geneticengine.metahandlers.lists import ListSizeBetween
from geneticengine.algorithms.gp import GP


map = """.###............................
...#............................
...#.....................###....
...#....................#....#..
...#....................#....#..
...####.#####........##.........
............#................#..
............#.......#...........
............#.......#........#..
............#.......#...........
....................#...........
............#................#..
............#...................
............#.......#.....###...
............#.......#..#........
.................#..............
................................
............#...........#.......
............#...#..........#....
............#...#...............
............#...#...............
............#...#.........#.....
............#..........#........
............#...................
...##. .#####....#...............
.#..............#...............
.#..............#...............
.#......#######.................
.#.....#........................
.......#........................
..####..........................
................................"""


class Action(Protocol):
    pass


@dataclass
class ActionBlock(Node, Action):
    actions: Annotated[List[Action], ListSizeBetween(2, 3)]

    def __init__(self,actions):
        self.actions = actions


@dataclass
class IfFood(Node, Action):
    yes_food: Action
    no_food: Action

    def __init__(self,yes_food,no_food):
        self.yes_food = yes_food
        self.no_food = no_food


@dataclass
class Move(Node, Action):
    pass


@dataclass
class Right(Node, Action):
    pass


@dataclass
class Left(Node, Action):
    pass


class Direction(Enum):
    EAST = 0
    SOUTH = 1
    WEST = 2
    NORTH = 3


class Position(Enum):
    EMPTY = 0
    FOOD = 1


def map_from_string(map_str: str) -> List[List[Position]]:
    return [
        [pos == "#" and Position.FOOD or Position.EMPTY for pos in line]
        for line in map_str.split("\n")
    ]


def next_pos(
    pos: Tuple[int, int, Direction], map: List[List[Position]]
) -> Tuple[int, int]:
    masks = {
        Direction.EAST: (0, 1),
        Direction.SOUTH: (1, 0),
        Direction.WEST: (0, -1),
        Direction.NORTH: (-1, 0),
    }
    row = (pos[0] + masks[pos[2]][0]) % len(map)
    col = (pos[1] + masks[pos[2]][1]) % len(map[0])
    return (row, col)


def food_in_front(pos: Tuple[int, int, Direction], map: List[List[Position]]) -> bool:
    (row, col) = next_pos(pos, map)
    return map[row][col] == Position.FOOD


def simulate(a: Action, map_str: str) -> int:
    next_instructions = [a]
    food_consumed = 0
    map = map_from_string(map_str)
    current_pos: Tuple[int, int, Direction] = (
        0,
        0,
        Direction.EAST,
    )  # row, col, direction
    while next_instructions:
        current_instruction = next_instructions.pop()
        if isinstance(current_instruction, ActionBlock):
            next_instructions = current_instruction.actions + next_instructions
        elif isinstance(current_instruction, IfFood):
            if food_in_front(current_pos, map):
                next_instructions.insert(0, current_instruction.yes_food)
            else:
                next_instructions.insert(0, current_instruction.no_food)
        elif isinstance(current_instruction, Move):
            (row, col) = next_pos(current_pos, map)
            if map[row][col] == Position.FOOD:
                food_consumed += 1
                map[row][col] = Position.EMPTY
            current_pos = (row, col, current_pos[2])
        elif isinstance(current_instruction, Left):
            current_pos = (
                current_pos[0],
                current_pos[1],
                Direction((current_pos[2].value - 1) % 4),
            )
        elif isinstance(current_instruction, Right):
            current_pos = (
                current_pos[0],
                current_pos[1],
                Direction((current_pos[2].value + 1) % 4),
            )
    return food_consumed


if __name__ == "__main__":
    g = extract_grammar([ActionBlock, IfFood, Move, Right, Left], Action)
    alg = GP(
        g,
        treebased_representation,
        lambda p: simulate(p, map),
        minimize=False,
        max_depth=10,
        number_of_generations=50,
        population_size=1500,
        novelty=50,
    )
    (b, bf) = alg.evolve()
    print(bf, b)
