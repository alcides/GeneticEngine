from __future__ import annotations

from typing import Any
from typing import Generic
from typing import TypeVar

from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import Source
from geneticengine.core.tree import TreeNode

g = TypeVar("g")
p = TypeVar("p")


class Representation(Generic[g, p]):
    grammar: Grammar
    min_depth: int
    max_depth: int

    def __init__(self, grammar: Grammar, max_depth: int):
        self.grammar = grammar
        self.min_depth = self.grammar.distanceToTerminal[self.grammar.starting_symbol]
        self.max_depth = max_depth
        assert self.min_depth <= self.max_depth

    def create_individual(self, r: Source, depth: int | None = None, **kwargs) -> g:
        ...

    def mutate_individual(
        self,
        r: Source,
        ind: g,
        depth: int,
        ty: type,
        specific_type: type = None,
        depth_aware_mut: bool = False,
        **kwargs,
    ) -> g:
        ...

    def crossover_individuals(
        self,
        r: Source,
        i1: g,
        i2: g,
        depth: int,
        specific_type: type = None,
        depth_aware_co: bool = False,
        **kwargs,
    ) -> tuple[g, g]:
        ...

    def genotype_to_phenotype(self, genotype: g) -> p:
        ...

    def phenotype_to_genotype(self, phenotype: p) -> g:
        """Takes an existing program and adapts it to be used in the right
        representation."""
        ...
