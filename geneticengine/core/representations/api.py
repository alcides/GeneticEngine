from typing import Any, Callable, Generic, Type, TypeVar, Tuple
from dataclasses import dataclass
from geneticengine.core.random.sources import Source
from geneticengine.core.grammar import Grammar
from geneticengine.core.tree import TreeNode

g = TypeVar("g")


class Representation(Generic[g]):
    def create_individual(self, r: Source, g: Grammar, depth: int) -> g:
        ...

    def mutate_individual(self, r: Source, g: Grammar, ind: g, depth: int, ty: Type) -> g:
        ...

    def crossover_individuals(
        self, r: Source, g: Grammar, i1: g, i2: g, int
    ) -> Tuple[g, g]:
        ...

    def genotype_to_phenotype(self, g: Grammar, genotype: g) -> TreeNode:
        ...
