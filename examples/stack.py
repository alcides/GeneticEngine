from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Annotated

from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.evaluation.budget import TimeBudget
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.representations.stackgggp import StackBasedGGGPRepresentation


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


def fitness_function(n):
    return 0


def main(seed=123):
    grammar = extract_grammar([Value, ScalarVar, VectorialVar, Mean, CumulativeSum], Scalar)
    prob = SingleObjectiveProblem(
        fitness_function=fitness_function,
        minimize=True,
        target=0,
    )

    alg = GeneticProgramming(
        problem=prob,
        budget=TimeBudget(3),
        population_size=20,
        representation=StackBasedGGGPRepresentation(grammar, 2048),
        random=NativeRandomSource(seed),
    )
    best = alg.search()[0]
    print(
        f"Fitness of {best.get_fitness(alg.get_problem())} by genotype: {best.genotype} with phenotype: {best.get_phenotype()}",
    )


if __name__ == "__main__":
    main()
