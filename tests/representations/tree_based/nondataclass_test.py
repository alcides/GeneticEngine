from __future__ import annotations

from geneticengine.grammar.decorators import abstract
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import MaxDepthDecider
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
    d = MaxDepthDecider(r, g, max_depth=20)
    x = random_node(r, g, Node, d)
    for i in range(100):
        y = random_node(r, g, Node, d)
        x, _ = tree_crossover(r, g, x, y, d)


def test_mutation():
    r = NativeRandomSource(seed=1)
    g = extract_grammar([Leaf, Inner], Node)
    d = MaxDepthDecider(r, g, max_depth=3)
    x = random_node(r, g, Node, d)
    for i in range(100):
        x = tree_mutate(r, g, x, Node, d)
