from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Generic
from typing import Tuple
from typing import Type
from typing import TypeVar

from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.tree.initialization_methods import Initialization_Method
from geneticengine.core.tree import TreeNode

g = TypeVar("g")


class Representation(Generic[g]):
    depth: int
    method: Initialization_Method
    
    @abstractmethod
    def create_individual(self, r: Source, g: Grammar, depth: int) -> g:
        ...

    @abstractmethod
    def mutate_individual(
        self,
        r: Source,
        g: Grammar,
        ind: g,
        depth: int,
        ty: type,
        specific_type: type | None = None,
        depth_aware_mut: bool = False,
    ) -> g:
        ...

    @abstractmethod
    def crossover_individuals(
        self,
        r: Source,
        g: Grammar,
        i1: g,
        i2: g,
        int,
        specific_type: type | None = None,
        depth_aware_co: bool = False,
    ) -> tuple[g, g]:
        ...

    @abstractmethod
    def genotype_to_phenotype(self, g: Grammar, genotype: g) -> TreeNode:
        ...
