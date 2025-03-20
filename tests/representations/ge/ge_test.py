from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
from typing import Annotated, Any
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.solutions.individual import PhenotypicIndividual

from geneticengine.grammar.grammar import Grammar, extract_grammar
from geneticengine.random.sources import NativeRandomSource, RandomSource
from geneticengine.representations.grammatical_evolution.ge import GrammaticalEvolutionRepresentation
from geneticengine.grammar.metahandlers.base import MetaHandlerGenerator


class RandomDNA(MetaHandlerGenerator):
    nucleotides: list[str]

    def __init__(self, size: int):
        self.size = size
        self.nucleotides = ["A", "C", "G", "T"]

    def generate(
        self,
        random: RandomSource,
        grammar: Grammar,
        base_type: type,
        rec: Any,
        dependent_values: dict[str, Any],
        parent_values: list[dict[str,Any]],
    ):
        sequence = "".join(random.choice(self.nucleotides) for _ in range(self.size))
        return sequence

    def validate(self, v) -> bool:
        return len(v) == self.size and all(x in self.nucleotides for x in v)


class Root(ABC):
    pass


@dataclass
class Leaf(Root):
    letters: Annotated[str, RandomDNA(size=3)]


def test_metahandler_gen():
    r = NativeRandomSource(seed=1)
    g = extract_grammar([Leaf], Root)
    d = MaxDepthDecider(r, g, 2)
    rep = GrammaticalEvolutionRepresentation(g, d)
    ind = PhenotypicIndividual(genotype=rep.create_genotype(random=r), representation=rep)

    assert ind.get_phenotype()
