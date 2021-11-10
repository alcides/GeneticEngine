from typing import Any, Callable, Generic, TypeVar, Tuple
from dataclasses import dataclass
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.grammar import Grammar

g = TypeVar("g")
p = TypeVar("p")


@dataclass
class Representation(Generic[g, p]):
    create_individual: Callable[[RandomSource, Grammar, int], g]
    mutate_individual: Callable[[RandomSource, Grammar, g, int], g]
    crossover_individuals: Callable[[RandomSource, Grammar, g, g, int], Tuple[g, g]]
    genotype_to_phenotype: Callable[[g], p]
