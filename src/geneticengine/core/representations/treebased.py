import sys

from typing import Annotated, Any

from geneticengine.core.random.sources import RandomSource
from geneticengine.core.grammar import Grammar
from geneticengine.core.utils import get_arguments
from geneticengine.exceptions import GeneticEngineError


def isTerminal(p: type) -> bool:
    for (a, at) in get_arguments(p):
        if hasattr(at, "__metadata__"):
            continue
        else:
            return False
    return True


def random_individual(
    r: RandomSource, g: Grammar, depth: int = 5, starting_symbol: Any = None
):
    if depth < 0:
        raise GeneticEngineError("Recursion Depth reached")

    if starting_symbol is None:
        starting_symbol = g.starting_symbol

    if starting_symbol is int:
        return r.randint(-(sys.maxsize - 1), sys.maxsize)
    if hasattr(starting_symbol, "__metadata__"):
        metahandler = starting_symbol.__metadata__[0]
        return metahandler.generate(r)
    if starting_symbol not in g.productions:
        raise GeneticEngineError(f"Symbol {starting_symbol} not in grammar rules.")

    valid_productions = g.productions[starting_symbol]
    if depth <= 1:
        valid_productions = [vp for vp in valid_productions if isTerminal(vp)]
    if not valid_productions:
        raise GeneticEngineError(f"No productions for non-terminal {starting_symbol}")
    rule = r.choice(valid_productions)
    args = [random_individual(r, g, depth - 1, at) for (a, at) in get_arguments(rule)]

    return rule(*args)