from abc import ABC
from dataclasses import dataclass

from geneticengine.algorithms.enumerative import iterate_individuals
from geneticengine.grammar.grammar import extract_grammar


class Root(ABC):
    pass


@dataclass
class Leaf(Root):
    pass


@dataclass
class Branch1(Root):
    v1: Root
    v2: Root


@dataclass
class Branch2(Root):
    v1: Root
    v2: Root


def test_enumerative():
    g = extract_grammar([Leaf, Branch1, Branch2], Root)
    exp = [Leaf(), Branch1(Leaf(), Leaf()), Branch2(Leaf(), Leaf())]

    xs = []
    for x in iterate_individuals(g, Root):
        xs.append(x.instance)
        if len(xs) > 10:
            break
    print(xs)

    for expected, real in zip(exp, xs):
        assert expected == real
