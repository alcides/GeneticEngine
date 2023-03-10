from __future__ import annotations

from geneticengine.core.decorators import abstract
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.tree.treebased import (
    crossover as tree_crossover,
)
from geneticengine.core.representations.tree.treebased import mutate
from geneticengine.core.representations.tree.treebased import random_node


@abstract
class Node:
    pass


class Leaf(Node):
    pass


class Inner(Node):
    def __init__(self, left: Node, right: Node):
        self.l = left
        self.r = right


def test_crossover():
    r = RandomSource(seed=1)
    g: Grammar = extract_grammar([Leaf, Inner], Node)
    x = random_node(r, g, 20, Node)
    for i in range(100):
        y = random_node(r, g, 20, Node)
        x, _ = tree_crossover(r, g, x, y, 20)


def test_mutation():
    r = RandomSource(seed=1)
    g: Grammar = extract_grammar([Leaf, Inner], Node)
    x = random_node(r, g, 20, Node)
    for i in range(100):
        x = mutate(r, g, x, 20, Node)
