from __future__ import annotations

from geneticengine.grammar.decorators import abstract
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.treebased import tree_crossover
from geneticengine.representations.tree.treebased import tree_mutate
from geneticengine.representations.tree.treebased import random_node


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
    r = NativeRandomSource(seed=1)
    g = extract_grammar([Leaf, Inner], Node)
    x = random_node(r, g, 20, Node)
    for i in range(100):
        y = random_node(r, g, 20, Node)
        x, _ = tree_crossover(r, g, x, y, 20)


def test_mutation():
    r = NativeRandomSource(seed=1)
    g = extract_grammar([Leaf, Inner], Node)
    x = random_node(r, g, 20, Node)
    for i in range(100):
        x = tree_mutate(r, g, x, 20, Node)
