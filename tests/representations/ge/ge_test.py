from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
from typing import Annotated
from geneticengine.solutions.individual import Individual

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
        r: RandomSource,
        g: Grammar,
        rec,
        new_symbol,
        depth: int,
        base_type,
        context: dict[str, str],
    ):
        sequence = "".join(r.choice(self.nucleotides) for _ in range(self.size))
        rec(sequence)


class Root(ABC):
    pass


@dataclass
class Leaf(Root):
    letters: Annotated[str, RandomDNA(size=3)]


def test_metahandler_gen():
    r = NativeRandomSource(seed=1)
    g = extract_grammar([Leaf], Root)
    rep = GrammaticalEvolutionRepresentation(g, max_depth=2)
    ind = Individual(genotype=rep.create_individual(r=r, g=g), genotype_to_phenotype=rep.genotype_to_phenotype)

    assert ind.get_phenotype()
