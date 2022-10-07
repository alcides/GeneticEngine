from __future__ import annotations

import csv
import time
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from optparse import OptionParser
from typing import Annotated
from typing import List
from typing import Tuple

import global_vars as gv

from geneticengine.algorithms.gp.gp import GP
from geneticengine.algorithms.hill_climbing import HC
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.grammatical_evolution.dynamic_structured_ge import (
    dsge_representation,
)
from geneticengine.core.representations.grammatical_evolution.ge import (
    ge_representation,
)
from geneticengine.core.representations.grammatical_evolution.structured_ge import (
    sge_representation,
)
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.metahandlers.lists import ListSizeBetween


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


class ActionMain(ABC):
    pass


class Action(ABC):
    pass


@dataclass
class ActionBlock(ActionMain):
    actions: Annotated[list[Action], ListSizeBetween(2, 30)]


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


def map_from_string(map_str: str) -> list[list[Position]]:
    return [
        [pos == "#" and Position.FOOD or Position.EMPTY for pos in line]
        for line in map_str.split("\n")
    ]


def next_pos(
    pos: tuple[int, int, Direction],
    map: list[list[Position]],
) -> tuple[int, int]:
    masks = {
        Direction.EAST: (0, 1),
        Direction.SOUTH: (1, 0),
        Direction.WEST: (0, -1),
        Direction.NORTH: (-1, 0),
    }
    row = (pos[0] + masks[pos[2]][0]) % len(map)
    col = (pos[1] + masks[pos[2]][1]) % len(map[0])
    return (row, col)


def food_in_front(pos: tuple[int, int, Direction], map: list[list[Position]]) -> bool:
    (row, col) = next_pos(pos, map)
    return map[row][col] == Position.FOOD


def simulate(a: Action, map_str: str) -> int:
    next_instructions: list[Action] = [a]
    food_consumed = 0
    map = map_from_string(map_str)
    current_pos: tuple[int, int, Direction] = (
        0,
        0,
        Direction.EAST,
    )  # row, col, direction
    while next_instructions:
        current_instruction = next_instructions.pop(0)  # Default is -1
        if isinstance(
            current_instruction,
            ActionBlock,
        ):  # ActionBlock contains list of action lists.
            for action in reversed(current_instruction.actions):
                next_instructions = [action] + next_instructions
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


def preprocess():
    return extract_grammar([ActionBlock, Action, IfFood, Move, Right, Left], ActionMain)


fitness_function = lambda p: simulate(p, map)


def evolve(
    seed,
    mode,
    save_to_csv: str = None,
    representation="treebased_representation",
):
    if representation == "ge":
        representation = ge_representation
    elif representation == "sge":
        representation = sge_representation
    elif representation == "dsge":
        representation = dsge_representation
    else:
        representation = treebased_representation

    g = preprocess()
    alg = GP(
        g,
        fitness_function,
        representation=representation,
        probability_crossover=gv.PROBABILITY_CROSSOVER,
        probability_mutation=gv.PROBABILITY_MUTATION,
        number_of_generations=gv.NUMBER_OF_GENERATIONS,
        max_depth=gv.MAX_DEPTH,
        population_size=gv.POPULATION_SIZE,
        selection_method=gv.SELECTION_METHOD,
        n_elites=gv.N_ELITES,
        # ----------------
        minimize=False,
        seed=seed,
        timer_stop_criteria=mode,
        save_to_csv=save_to_csv,
        save_genotype_as_string=False,
    )
    (b, bf, bp) = alg.evolve(verbose=1)
    return b, bf


if __name__ == "__main__":
    representations = ["treebased_representation", "ge", "dsge"]

    parser = OptionParser()
    parser.add_option("-s", "--seed", dest="seed", type="int")
    parser.add_option("-r", "--representation", dest="representation", type="int")
    parser.add_option(
        "-t",
        "--timed",
        dest="timed",
        action="store_const",
        const=True,
        default=False,
    )
    (options, args) = parser.parse_args()

    timed = options.timed
    seed = options.seed
    example_name = __file__.split(".")[0].split("\\")[-1].split("/")[-1]
    representation = representations[options.representation]
    print(seed, example_name, representation)
    evol_method = evolve

    mode = "generations"
    if timed:
        mode = "time"
    dest_file = f"{gv.RESULTS_FOLDER}/{mode}/{example_name}/{representation}/{seed}.csv"

    start = time.time()
    b, bf = evolve(seed, timed, dest_file, representation)
    end = time.time()
    csv_row = [mode, example_name, representation, seed, bf, (end - start), b]
    with open(
        f"./{gv.RESULTS_FOLDER}/{mode}/{example_name}/{representation}/main.csv",
        "a",
        newline="",
    ) as outfile:
        writer = csv.writer(outfile)
        writer.writerow(csv_row)
