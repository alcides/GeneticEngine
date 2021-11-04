from dataclasses import dataclass
from enum import Enum
from typing import Annotated, List, Tuple
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.treebased import treebased_representation
from geneticengine.metahandlers.lists import ListSizeBetween
from geneticengine.algorithms.gp.gp import GP


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


class Action:
    pass


@dataclass
class ActionBlock(Action):
    actions: Annotated[List[Action], ListSizeBetween(2, 3)]


@dataclass
class IfFood(Action):
    yes_food: Action
    no_food: Action


@dataclass
class Move(Action):
    pass


@dataclass
class Right(Action):
    pass


@dataclass
class Left(Action):
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
        current_instruction = next_instructions.pop(0)  # Default is -1
        if isinstance(
            current_instruction, ActionBlock
        ):  # ActionBlock contains list of action lists.
            for action in reversed(current_instruction.actions):
                next_instructions = action + next_instructions
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
        max_depth=6,
        number_of_generations=50,
        population_size=1500,
        n_novelties=50,
    )
    (b, bf) = alg.evolve()
    print(bf, b)
