from geneticengine.core.decorators import abstract
from geneticengine.core.grammar import Grammar, extract_grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.tree.treebased import (
    random_node,
    PI_Grow,
    tree_crossover,
    Grow,
    mutate,
)


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
    x = random_node(r, g, 20, Node, method=PI_Grow)
    for i in range(100):
        y = random_node(r, g, 20, Node, method=PI_Grow)
        x, _ = tree_crossover(r, g, x, y, 20)


def test_mutation():
    r = RandomSource(seed=1)
    g: Grammar = extract_grammar([Leaf, Inner], Node)
    x = random_node(r, g, 20, Node, method=PI_Grow)
    for i in range(100):
        x = mutate(r, g, x, 20, Node)
