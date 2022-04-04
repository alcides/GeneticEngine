from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Generic
from typing import Tuple
from typing import Type
from typing import TypeVar

from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import Source
from geneticengine.core.tree import TreeNode

g = TypeVar("g")


class Representation(Generic[g]):
    def create_individual(self, r: Source, g: Grammar, depth: int) -> g:
        ...

    def mutate_individual(
        self,
        r: Source,
        g: Grammar,
        ind: g,
        depth: int,
        ty: type,
    ) -> g:
        ...

    def crossover_individuals(
        self,
        r: Source,
        g: Grammar,
        i1: g,
        i2: g,
        int,
    ) -> tuple[g, g]:
        ...

    def genotype_to_phenotype(self, g: Grammar, genotype: g) -> TreeNode:
        ...
