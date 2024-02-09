from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Annotated

from geneticengine.grammar.grammar import extract_grammar
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.stackgggp import StackBasedGGGPRepresentation
from geneticengine.grammar.metahandlers.lists import ListSizeBetween


class Root(ABC):
    pass


@dataclass
class Leaf(Root):
    pass


@dataclass
class MiddleList(Root):
    z: Annotated[list[Root], ListSizeBetween(2, 3)]


@dataclass
class Concrete(Root):
    x: int


@dataclass
class Middle(Root):
    x: Root


class TestStackBased:
    def test_stack_generation(self):
        r = NativeRandomSource(seed=1)
        g = extract_grammar([Concrete, Middle, MiddleList], Root, False)

        max_depth = 10

        repr = StackBasedGGGPRepresentation(g, max_depth)
        genotype = repr.instantiate(r, max_depth)

        for i in range(10):
            genotype = repr.mutate(genotype, None, None, repr, r, 0, i)

        phenotype = repr.map(genotype)
        assert phenotype is not None
