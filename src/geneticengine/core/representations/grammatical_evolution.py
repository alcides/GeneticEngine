from typing import Any, List, Protocol

from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.base import Representation


class Individual(Protocol):
    pass


def random_individual(
    r: RandomSource, g: Grammar, depth: int = 5, starting_symbol: Any = None
):
    return [r.randint(0, 10000) for _ in range(256)]


def mutate(r: RandomSource, g: Grammar, ind: List[int]) -> List[int]:
    rindex = r.randint(0, 255)
    clone = [i for i in ind]
    clone[rindex] = r.randint(0, 10000)
    return clone


ge_representation = Representation(
    create_individual=random_individual,
    mutate_individual=mutate,
    crossover_individuals=tree_crossover,
    preprocess_grammar=preprocess_grammar,
)
