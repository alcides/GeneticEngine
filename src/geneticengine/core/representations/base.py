from typing import Any, Callable, Generic, TypeVar, Tuple
from dataclasses import dataclass
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.grammar import Grammar
from geneticengine.core.tree import TreeNode

g = TypeVar("g")


@dataclass
class Representation(Generic[g]):
    create_individual: Callable[[RandomSource, Grammar, int], g]
    mutate_individual: Callable[[RandomSource, Grammar, g, int], g]
    crossover_individuals: Callable[[RandomSource, Grammar, g, g, int], Tuple[g, g]]
    genotype_to_phenotype: Callable[[Grammar, g], TreeNode]
