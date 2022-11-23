from __future__ import annotations
from geneticengine.core.representations.tree.treebased import treebased_representation

from abc import ABC
from dataclasses import dataclass
from typing import Annotated

from geneticengine.core.grammar import extract_grammar
from geneticengine.core.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.metahandlers.ints import IntRange

from geneticengine.algorithms.gp.gp import GP


class Scalar(ABC):
    pass


class Vectorial(ABC):
    pass


@dataclass
class Value(Scalar):
    value: float


@dataclass
class ScalarVar(Scalar):
    index: Annotated[int, IntRange(0, 1)]


@dataclass
class VectorialVar(Vectorial):
    index: Annotated[int, IntRange(2, 3)]


@dataclass
class CumulativeSum(Vectorial):
    arr: Vectorial


@dataclass
class Mean(Scalar):
    arr: Vectorial


@dataclass
class Add(Scalar):
    right: Scalar
    left: Scalar


g = extract_grammar([Value, ScalarVar, VectorialVar, Mean, CumulativeSum], Scalar)
print(g)


def fitness_function(n):
    return 0


def evolve(g, seed, mode):
    alg = GP(
        g,
        representation=treebased_representation,
        problem=SingleObjectiveProblem(
            fitness_function=fitness_function,
            minimize=True,
            target_fitness=0,
        ),
        population_size=20,
        number_of_generations=5,
        timer_stop_criteria=mode,
        seed=seed,
    )
    (b, bf, bp) = alg.evolve()
    return b, bf


if __name__ == "__main__":
    bf, b = evolve(g, 0, False)
    print(b)
    print(f"With fitness: {bf}")
