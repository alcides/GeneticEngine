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

        repr = StackBasedGGGPRepresentation(g, 2048)
        genotype = repr.create_genotype(r)

        for i in range(10):
            genotype = repr.mutate(r, genotype)

        phenotype = repr.genotype_to_phenotype(genotype)
        assert phenotype is not None
