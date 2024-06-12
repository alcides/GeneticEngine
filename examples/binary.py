from abc import ABC
import copy
from dataclasses import dataclass
from typing import Annotated, Any

from geneticengine.grammar import Grammar
from geneticengine.evaluation.budget import AnyOf, TargetFitness, TimeBudget
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource, RandomSource

from geneticengine.grammar.metahandlers.lists import ListSizeBetween
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.representations.tree.treebased import (
    TreeBasedRepresentation,
    random_node,
)
from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.representations.tree.utils import relabel_nodes_of_trees
from geneticengine.solutions.tree import TreeNode


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


def ensure_tree_node(x: Any, grammar: Grammar) -> TreeNode:
    x.gengy_labeled = True
    x.gengy_distance_to_term = 1
    x.gengy_nodes = 4
    x.gengy_weighted_nodes = 5
    x.gengy_types_this_way = {}
    x.gengy_init_values = (1,)
    relabel_nodes_of_trees(x, g)
    return x


class BinaryListTreeBasedRepresentation(TreeBasedRepresentation):
    def __init__(self, grammar, max_depth):
        super().__init__(grammar, max_depth)

    def mutate(self, random: RandomSource, internal: TreeNode, **kwargs) -> TreeNode:
        assert isinstance(internal, BinaryList)

        random_pos = random.randint(0, SIZE - 1)
        next_val = random_node(random=r, grammar=g, max_depth=1, starting_symbol=Bit)
        c = copy.deepcopy(internal)
        c.byte[random_pos] = next_val
        return c

    def crossover(
        self,
        random: RandomSource,
        parent1: TreeNode,
        parent2: TreeNode,
        **kwargs,
    ) -> tuple[TreeNode, TreeNode]:
        assert isinstance(parent1, BinaryList)
        assert isinstance(parent2, BinaryList)
        p = random.randint(0, len(parent1.byte))
        p1 = copy.deepcopy(parent1.byte[:p])
        q1 = copy.deepcopy(parent2.byte[:p])
        p2 = copy.deepcopy(parent2.byte[: len(parent2.byte) - p])
        q2 = copy.deepcopy(parent1.byte[: len(parent1.byte) - p])
        b1 = ensure_tree_node(BinaryList(byte=p1 + p2), self.grammar)
        b2 = ensure_tree_node(BinaryList(byte=q1 + q2), self.grammar)
        assert isinstance(b1, TreeNode)
        assert isinstance(b2, TreeNode)
        return (b1, b2)


if __name__ == "__main__":
    r = NativeRandomSource()
    g = extract_grammar([One, Zero, BinaryList], BinaryList)
    repr = BinaryListTreeBasedRepresentation(grammar=g, max_depth=4)

    gp = GeneticProgramming(
        problem=SingleObjectiveProblem(fitness_function=fitness, minimize=False),
        budget=AnyOf(TargetFitness(1), TimeBudget(3)),
        representation=repr,
        random=r,
        population_size=10,
    )
    out = gp.search()
    print(out)
