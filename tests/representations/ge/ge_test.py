from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
from typing import Annotated, Any
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
        random: RandomSource,
        grammar: Grammar,
        base_type: type,
        rec: Any,
        dependent_values: dict[str, Any],
    ):
        sequence = "".join(random.choice(self.nucleotides) for _ in range(self.size))
        return sequence


class Root(ABC):
    pass


@dataclass
class Leaf(Root):
    letters: Annotated[str, RandomDNA(size=3)]


def test_metahandler_gen():
    r = NativeRandomSource(seed=1)
    g = extract_grammar([Leaf], Root)
    rep = GrammaticalEvolutionRepresentation(g, max_depth=2)
    ind = Individual(genotype=rep.create_genotype(random=r), representation=rep)

    assert ind.get_phenotype()
