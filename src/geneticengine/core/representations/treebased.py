import sys

from typing import Annotated, Any, TypeVar

from geneticengine.core.random.sources import RandomSource
from geneticengine.core.grammar import Grammar
from geneticengine.core.representations.base import Representation
from geneticengine.core.tree import Node
from geneticengine.core.utils import get_arguments, isTerminal
from geneticengine.exceptions import GeneticEngineError


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
    node = rule(*args)
    node.depth = max([1] + [n.depth for n in args if hasattr(n, "depth")])
    node.nodes = 1 + sum([n.nodes for n in args if hasattr(n, "nodes")])
    return node


def mutate(r: RandomSource, g: Grammar, i: Node) -> Node:
    c = r.randint(0, i.nodes)
    if c == 0:
        ty = i.__class__.__bases__[1]
        replacement = random_individual(r, g, i.depth + 1, ty)
        return replacement
    else:
        for field in i.__annotations__:
            if hasattr(i.__annotations__[field], "nodes"):
                count = getattr(i, field).nodes
                if c < count:
                    setattr(i, field, mutate(r, g, getattr(i, field)))
                    break
                else:
                    c -= count
        return i


treebased_representation = Representation(
    create_individual=random_individual,
    mutate_individual=mutate,
    crossover_individuals=lambda x, y, a, b: (a, b),
)
