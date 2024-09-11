from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import sys
from typing import TypeVar
from geneticengine.exceptions import GeneticEngineError

from geneticengine.grammar.grammar import Grammar
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import (
    RepresentationWithCrossover,
    RepresentationWithMutation,
    Representation,
)
from geneticengine.representations.tree.initializations import SynthesisDecider
from geneticengine.representations.tree.treebased import random_tree
from geneticengine.solutions.tree import LocalSynthesisContext, TreeNode


T = TypeVar("T")

MAX_GENE_VALUE = 1024


@dataclass
class Genotype:
    random: RandomSource
    dna: dict[type, list[int]]

    def get(self, ty: type, n: int):
        if ty not in self.dna:
            self.dna[ty] = []
        while len(self.dna[ty]) <= n:
            nvalue = self.random.randint(0, MAX_GENE_VALUE)
            self.dna[ty].append(nvalue)
        return self.dna[ty][n]


class DynamicSGEDecider(SynthesisDecider):
    def __init__(self, genotype: Genotype, grammar: Grammar, max_depth: int = 10, max_string_length: int = 128):
        self.genotype = genotype
        self.grammar = grammar
        self.max_depth = max_depth
        self.positions: dict[type, int] = {t: 0 for t in grammar.all_nodes}
        self.max_string_length = max_string_length
        self.validate()

    def read(self, ty):
        v = self.genotype.get(ty, self.positions[ty])
        self.positions[ty] += 1
        return v

    def random_int(self, min_int=-sys.maxsize, max_int=sys.maxsize) -> int:
        v = self.read(int)
        return v % (max_int - min_int) + min_int

    def random_float(self) -> float:
        max_float = sys.float_info.max
        min_float = -sys.float_info.max
        v = self.read(float)
        return v % (max_float - min_float) + min_float

    def random_chr(self) -> int:
        return self.random_int(32, 128)

    def random_str(self) -> str:
        length = self.random_int(0, self.max_string_length)
        return str(self.random_chr() for _ in range(length))

    def random_bool(self) -> bool:
        return self.read(bool)

    def choose_production_alternatives(self, ty: type, alternatives: list[type], ctx: LocalSynthesisContext) -> type:
        assert len(alternatives) > 0, "No alternatives presented"

        v = self.read(ty)
        alternatives = [
            x for x in alternatives if self.grammar.get_distance_to_terminal(x) <= (self.max_depth - ctx.depth)
        ]
        return alternatives[v % len(alternatives)]

    def choose_options(self, alternatives: list[T], ctx: LocalSynthesisContext) -> T:
        assert len(alternatives) > 0, "No alternatives presented"
        v = self.read(Genotype)  # We use Genotype as the infrastructure
        return alternatives[v % len(alternatives)]

    def validate(self) -> None:
        if self.max_depth <= self.grammar.get_min_tree_depth():
            if self.grammar.get_min_tree_depth() == 1000000:
                raise GeneticEngineError(
                    f"""Grammar's minimal tree depth is {self.grammar.get_min_tree_depth()}, which is the default tree depth.
                    It's highly like that there are nodes of your grammar than cannot reach any terminal.""",
                )
            raise GeneticEngineError(
                f"""Cannot use complete grammar for individual creation. Max depth ({self.max_depth})
                is smaller than grammar's minimal tree depth ({self.grammar.get_min_tree_depth()}).""",
            )


class DynamicStructuredGrammaticalEvolutionRepresentation(
    Representation[Genotype, TreeNode],
    RepresentationWithMutation[Genotype],
    RepresentationWithCrossover[Genotype],
):
    """This version uses a list of lists of integers to represent individuals,
    based on non-terminal symbols."""

    def __init__(
        self,
        grammar: Grammar,
        max_depth: int,
    ):
        """
        Args:
            grammar (Grammar): The grammar to use in the mapping
            max_depth (int): the maximum depth when performing the mapping
                (e.g., pi_grow, full, grow)
        """
        self.grammar = grammar
        self.max_depth = max_depth

    def create_genotype(self, random: RandomSource, **kwargs) -> Genotype:
        return Genotype(random, {})

    def genotype_to_phenotype(self, genotype: Genotype) -> TreeNode:
        decider = DynamicSGEDecider(genotype, self.grammar, self.max_depth)
        return random_tree(genotype.random, self.grammar, decider)

    def mutate(self, random: RandomSource, genotype: Genotype, **kwargs) -> Genotype:
        dna = deepcopy(genotype.dna)
        alternatives = list(genotype.dna.keys())
        if alternatives:
            rkey = random.choice(alternatives)
            if genotype.dna[rkey]:
                rindex = random.randint(0, len(genotype.dna[rkey]) - 1)
                dna[rkey][rindex] = random.randint(0, sys.maxsize)
        return Genotype(genotype.random, dna)

    def crossover(
        self,
        random: RandomSource,
        parent1: Genotype,
        parent2: Genotype,
        **kwargs,
    ) -> tuple[Genotype, Genotype]:
        keys = parent1.dna.keys()

        mask = [(k, random.random_bool()) for k in keys]
        c1 = dict()
        c2 = dict()
        for k, b in mask:
            if b:
                c1[k] = deepcopy(parent1.dna.get(k, []))
                c2[k] = deepcopy(parent2.dna.get(k, []))
            else:
                c1[k] = deepcopy(parent2.dna.get(k, []))
                c2[k] = deepcopy(parent1.dna.get(k, []))
        return (Genotype(parent1.random, c1), Genotype(parent2.random, c2))
