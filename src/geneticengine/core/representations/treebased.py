import sys

from geneticengine.core.random.sources import RandomSource
from geneticengine.core.grammar import Grammar
from geneticengine.core.utils import get_arguments
from geneticengine.exceptions import GeneticEngineError


def random_individual(
    r: RandomSource, g: Grammar, depth: int = 5, starting_symbol=None
):
    if starting_symbol is None:
        starting_symbol = g.starting_symbol

    if starting_symbol is int:
        return r.randint(-(sys.maxsize - 1), sys.maxsize)

    if starting_symbol not in g.productions:
        raise GeneticEngineError(f"Symbol {starting_symbol} not in grammar rules.")

    valid_productions = g.productions[starting_symbol]
    rule = r.choice(valid_productions)
    args = [random_individual(r, g, depth - 1, at) for (a, at) in get_arguments(rule)]

    return rule(*args)