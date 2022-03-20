from __future__ import annotations

from geneticengine.core.decorators import abstract
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.tree.treebased import Grow
from geneticengine.core.representations.tree.treebased import mutate
from geneticengine.core.representations.tree.treebased import PI_Grow
from geneticengine.core.representations.tree.treebased import random_node
from geneticengine.core.representations.tree.treebased import tree_crossover


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
    g: Grammar = extract_grammar(Node, globals(), [Leaf, Inner])
    x = random_node(r, g, 20, Node, method=PI_Grow)
    for i in range(100):
        y = random_node(r, g, 20, Node, method=PI_Grow)
        x, _ = tree_crossover(r, g, x, y, 20)


def test_mutation():
    r = RandomSource(seed=1)
    g: Grammar = extract_grammar(Node, globals(), [Leaf, Inner])
    x = random_node(r, g, 20, Node, method=PI_Grow)
    for i in range(100):
        x = mutate(r, g, x, 20, Node)
