from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Annotated

from geneticengine.algorithms.gp.gp import GP
from geneticengine.algorithms.gp.operators.stop import (
    AnyOfStoppingCriterium,
    SingleFitnessTargetStoppingCriterium,
    GenerationStoppingCriterium,
)
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.grammar.metahandlers.ints import IntRange


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
    )

    stopping_criterium = AnyOfStoppingCriterium(
        GenerationStoppingCriterium(100),
        SingleFitnessTargetStoppingCriterium(0),
    )

    alg = GP(
        representation=TreeBasedRepresentation(grammar, 10),
        problem=prob,
        population_size=20,
        random_source=NativeRandomSource(seed),
        stopping_criterium=stopping_criterium,
    )
    best = alg.evolve()
    print(
        f"Fitness of {best.get_fitness(prob)} by genotype: {best.genotype} with phenotype: {best.get_phenotype()}",
    )


if __name__ == "__main__":
    main()
