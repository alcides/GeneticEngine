from dataclasses import dataclass
import sys
from copy import deepcopy

from typing import Annotated, Any, Dict, TypeVar, Tuple, List, _AnnotatedAlias, _GenericAlias

from geneticengine.core.random.sources import RandomSource
from geneticengine.core.grammar import Grammar
from geneticengine.core.representations.base import Representation
from geneticengine.core.tree import Node
from geneticengine.core.utils import get_arguments
from geneticengine.exceptions import GeneticEngineError


@dataclass
class ProcessedGrammar(object):
    grammar: Grammar
    distanceToTerminal: Dict[Node, int]


def random_individual(
    r: RandomSource, pg: ProcessedGrammar, max_depth: int = 5, starting_symbol: Any = None
):
    g = pg.grammar
    if max_depth < 0:
        raise GeneticEngineError("Recursion Depth reached")

    if starting_symbol is None:
        starting_symbol = g.starting_symbol

    if starting_symbol is int:
        return r.randint(-(sys.maxsize - 1), sys.maxsize)
    elif hasattr(starting_symbol, "__origin__"):
        if starting_symbol.__origin__ is list:
            size = r.randint(0, max_depth)
            return [
                random_individual(r, pg, max_depth, starting_symbol.__args__[0])
                for _ in range(size)
            ]
    if hasattr(starting_symbol, "__metadata__"):
        metahandler = starting_symbol.__metadata__[0]
        recursive_generator = lambda: random_individual(
            r, pg, max_depth, starting_symbol.__args__[0]
        )
        return metahandler.generate(r, recursive_generator)
    if starting_symbol not in g.productions:
        raise GeneticEngineError(f"Symbol {starting_symbol} not in grammar rules.")

    valid_productions = g.productions[starting_symbol]

    valid_productions = [vp for vp in valid_productions if pg.distanceToTerminal[vp] <= max_depth]
    if not valid_productions:
        raise GeneticEngineError("No productions for non-terminal node with type: {}.".format(starting_symbol))
    rule = r.choice(valid_productions)
    args = [random_individual(r, pg, max_depth - 1, at) for (a, at) in get_arguments(rule)]
    node = rule(*args)
    node = relabel_nodes_of_trees(node)
    return node


def mutate_inner(r: RandomSource, pg: ProcessedGrammar, i: Node, max_depth: int) -> Node:
    c = r.randint(0, i.nodes - 1)
    if c == 0:
        ty = i.__class__.__bases__[1]
        replacement = random_individual(r, pg, max_depth - i.depth + 1, ty)
        return replacement
    else:
        for field in i.__annotations__:
            child = getattr(i, field)
            if hasattr(child, "nodes"):
                count = child.nodes
                if c <= count:
                    setattr(i, field, mutate_inner(r, pg, child, max_depth))
                    return i
                else:
                    c -= count
        return i


def mutate(r: RandomSource, pg: ProcessedGrammar, i: Node, max_depth: int) -> Node:
    new_tree = mutate_inner(r, pg, deepcopy(i), max_depth)
    relabeled_new_tree = relabel_nodes_of_trees(new_tree)
    return relabeled_new_tree


def find_in_tree(ty: type, o: Node, max_depth: int):
    if ty in o.__class__.__bases__ and o.distance_to_term <= max_depth:
        yield o
    if hasattr(o, "__annotations__"):
        for field in o.__annotations__:
            child = getattr(o, field)
            yield from find_in_tree(ty, child, max_depth)


def tree_crossover_inner(
    r: RandomSource, pg: ProcessedGrammar, i: Node, o: Node, max_depth: int
) -> Node:
    c = r.randint(0, i.nodes - 1)
    if c == 0:
        ty = i.__class__.__bases__[1]
        replacement = r.choice(list(find_in_tree(ty, o, max_depth - i.depth + 1)))
        if replacement is None:
            replacement = random_individual(r, pg, max_depth - i.depth + 1, ty) 
        return replacement
    else:
        for field in i.__annotations__:
            child = getattr(i, field)
            if hasattr(child, "nodes"):
                count = getattr(i, field).nodes
                if c <= count:
                    setattr(i, field, tree_crossover_inner(r, pg, getattr(i, field), o, max_depth))
                    return i
                else:
                    c -= count
        return i

def tree_crossover(
    r: RandomSource, pg: ProcessedGrammar, p1: Node, p2: Node, max_depth: int
) -> Tuple[Node, Node]:
    '''
    Given the two input trees [p1] and [p2], the grammar and the random source, this function returns two trees that are created by crossing over [p1] and [p2]. The first tree returned has [p1] as the base, and the second tree has [p2] as a base.
    '''
    new_tree1 = tree_crossover_inner(r, pg, deepcopy(p1), deepcopy(p2), max_depth)
    relabeled_new_tree1 = relabel_nodes_of_trees(new_tree1)
    new_tree2 = tree_crossover_inner(r, pg, deepcopy(p2), deepcopy(p1), max_depth)
    relabeled_new_tree2 = relabel_nodes_of_trees(new_tree2)
    return relabeled_new_tree1,relabeled_new_tree2

def tree_crossover_single_tree(
    r: RandomSource, pg: ProcessedGrammar, p1: Node, p2: Node, max_depth: int
) -> Node:
    '''
    Given the two input trees [p1] and [p2], the grammar and the random source, this function returns one tree that is created by crossing over [p1] and [p2]. The tree returned has [p1] as the base.
    '''
    new_tree = tree_crossover_inner(r, pg, deepcopy(p1), deepcopy(p2), max_depth)
    relabeled_new_tree = relabel_nodes_of_trees(new_tree)
    return relabeled_new_tree

def relabel_nodes_of_trees(i: Node, depth: int = 1):
    # print("Node: {}, nodes: {}, distance_to_term: {}, depth: {}.".format(i,i.nodes,i.distance_to_term,i.depth))
    def relabel_nodes(i: Node, depth: int = 1):
        i.depth = depth
        nodess = [0]
        distance_to_terms = [0]
        if hasattr(i, "__annotations__"):
            for field in i.__annotations__:
                child = getattr(i, field)
                if type(child) not in [int,str]:
                    _, nodes, distance_to_term = relabel_nodes(child,depth + 1)
                    nodess.append(nodes)
                    distance_to_terms.append(distance_to_term)
        i.nodes = sum(nodess) + 1
        i.distance_to_term = max(distance_to_terms) + 1
        return i, i.nodes, i.distance_to_term
    # print("Node: {}, nodes: {}, distance_to_term: {}, depth: {}.".format(i,i.nodes,i.distance_to_term,i.depth))
    relabeled_tree, _, _ = relabel_nodes(i, depth)
    return relabeled_tree

def preprocess_grammar(g: Grammar) -> ProcessedGrammar:
    choice = set()
    for k in g.productions.keys():
        choice.add(k)
    sequence = set()
    for vv in g.productions.values():
        for v in vv:
            if v not in choice:
                sequence.add(v)
    all_sym = sequence.union(choice)
    dist_to_terminal = {int:1,str:1}
    for s in all_sym:
        dist_to_terminal[s] = 1000000
    changed = True
    while changed:
        changed = False
        for sym in all_sym:
            old_val = dist_to_terminal[sym]
            val = old_val
            if sym in choice:
                for prod in g.productions[sym]:
                    val = min(val, dist_to_terminal[prod])
            else:
                if hasattr(sym, "__annotations__"):
                    var = sym.__annotations__.values()
                    if isinstance(list(var)[0],_AnnotatedAlias):
                        t = list(var)[0].__origin__
                    else:
                        t = var.__iter__().__next__()
                    if isinstance(t,_GenericAlias):
                        t = t.__args__[0]
                    val = dist_to_terminal[t]
                    for prod in var:
                        if isinstance(prod,_AnnotatedAlias):
                            prod = prod.__origin__
                        if isinstance(prod,_GenericAlias):
                            prod = prod.__args__[0]
                        val = max(val, dist_to_terminal[prod]+1)
                else:
                    val = 1
            if val != old_val:
                changed = True
                dist_to_terminal[sym] = val


    return ProcessedGrammar(grammar=g, distanceToTerminal=dist_to_terminal)

treebased_representation = Representation(
    create_individual=random_individual,
    mutate_individual=mutate,
    crossover_individuals=tree_crossover,
    preprocess_grammar=preprocess_grammar
)
