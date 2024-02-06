from abc import ABC
from copy import deepcopy
import copy
from dataclasses import dataclass
from typing import Annotated
from geneticengine.algorithms.callbacks.callback import ProgressCallback
from geneticengine.algorithms.gp.operators.combinators import ParallelStep
from geneticengine.algorithms.gp.operators.crossover import GenericCrossoverStep
from geneticengine.algorithms.gp.operators.elitism import ElitismStep
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.evaluation import Evaluator
from geneticengine.problems import Problem, SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource, RandomSource
from geneticengine.representations.api import CrossoverOperator, MutationOperator, Representation

from geneticengine.grammar.metahandlers.lists import ListSizeBetween
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.representations.tree.treebased import (
    TreeBasedRepresentation,
    random_node,
)
from geneticengine.algorithms.gp.gp import GP
from geneticengine.algorithms.gp.operators.stop import SingleFitnessTargetStoppingCriterium, TimeStoppingCriterium
from geneticengine.algorithms.gp.operators.stop import AnyOfStoppingCriterium


SIZE = 50


class Bit(ABC):
    pass


class One(Bit):
    def __str__(self):
        return "1"


class Zero(Bit):
    def __str__(self):
        return "0"


@dataclass
class BinaryList:
    byte: Annotated[list[Bit], ListSizeBetween(SIZE, SIZE)]

    def __str__(self):
        return "".join(str(b) for b in self.byte)


def fitness(i: BinaryList):
    return str(i).count("1") / SIZE


if __name__ == "__main__":
    r = NativeRandomSource()
    g = extract_grammar([One, Zero, BinaryList], BinaryList)
    repr = TreeBasedRepresentation(grammar=g, max_depth=4)
    prob = SingleObjectiveProblem(
        fitness_function=fitness,
        minimize=False,
    )

    class BitFlip(MutationOperator[BinaryList]):
        def mutate(
            self,
            genotype: BinaryList,
            problem: Problem,
            evaluator: Evaluator,
            representation: Representation,
            random_source: RandomSource,
            index_in_population: int,
            generation: int,
        ) -> BinaryList:
            assert isinstance(genotype, BinaryList)
            random_pos = random_source.randint(0, SIZE - 1)
            next_val = random_node(r=random_source, g=g, max_depth=1, starting_symbol=Bit)
            copy = deepcopy(genotype)
            copy.byte[random_pos] = next_val
            return copy

    class SinglePointCrossover(CrossoverOperator[BinaryList]):
        def crossover(
            self,
            g1: BinaryList,
            g2: BinaryList,
            problem: Problem,
            representation: Representation,
            random_source: RandomSource,
            index_in_population: int,
            generation: int,
        ) -> tuple[BinaryList, BinaryList]:
            p = random_source.randint(0, len(g1.byte))
            p1 = copy.deepcopy(g1.byte[:p])
            q1 = copy.deepcopy(g2.byte[:p])
            p2 = copy.deepcopy(g2.byte[: len(g2.byte) - p])
            q2 = copy.deepcopy(g1.byte[: len(g1.byte) - p])
            return (BinaryList(byte=p1 + p2), BinaryList(byte=q1 + q2))

    step = ParallelStep(
        [
            GenericMutationStep(probability=0.5, operator=BitFlip()),
            GenericCrossoverStep(probability=0.5, operator=SinglePointCrossover()),
            ElitismStep(),
        ],
        weights=[5, 4, 1],
    )

    gp = GP(
        representation=repr,
        problem=prob,
        random_source=r,
        population_size=10,
        callbacks=[ProgressCallback()],
        stopping_criterium=AnyOfStoppingCriterium(SingleFitnessTargetStoppingCriterium(1.0), TimeStoppingCriterium(3)),
        step=step,
    )
    out = gp.evolve()
    print(out)
