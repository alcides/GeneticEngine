from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, TypeVar

from geneticengine.grammar.decorators import is_builtin
from geneticengine.grammar.grammar import Grammar
from geneticengine.solutions.tree import TreeNode
from geneticengine.grammar.utils import get_arguments
from geneticengine.grammar.utils import is_abstract
from geneticengine.grammar.utils import is_terminal
import dataclasses


def relabel_nodes(
    i: TreeNode,
    g: Grammar,
    is_list: bool = False,
) -> tuple[int, int, dict[type, list[Any]], int]:
    """Recomputes node specifics in the tree.\n Returns the number of nodes,
    distance to terminal (depth), typed this way, and the weighted number of
    nodes (counting depth points instead of nodes)."""
    non_terminals = g.non_terminals
    children: list[Any]
    if getattr(i, "gengy_labeled", False):
        return (
            i.gengy_nodes,
            i.gengy_distance_to_term,
            i.gengy_types_this_way,
            i.gengy_weighted_nodes,
        )
    number_of_nodes = 1
    distance_to_term = 1
    weighted_number_of_nodes = 0
    if is_list:
        number_of_nodes = 0
        distance_to_term = 0
    types_this_way = defaultdict(lambda: [])
    types_this_way[type(i)] = [i]

    if isinstance(i, list):
        children = [(type(obj), obj) for obj in i]
    elif dataclasses.is_dataclass(i):
        children = [(typ, getattr(i, aname)) for aname, typ in get_arguments(type(i))]
    elif is_builtin(type(i)):
        return (
            int(g.expansion_depthing),
            int(g.expansion_depthing),
            {type(i): [i]},
            int(g.expansion_depthing),
        )
    elif is_terminal(type(i), non_terminals):
        i.gengy_labeled = True
        i.gengy_distance_to_term = int(g.expansion_depthing)
        i.gengy_nodes = int(g.expansion_depthing)
        i.gengy_weighted_nodes = int(g.expansion_depthing)
        i.gengy_types_this_way = {type(i): [i]}
        return (
            int(g.expansion_depthing),
            int(g.expansion_depthing),
            {type(i): [i]},
            int(g.expansion_depthing),
        )
    elif hasattr(i, "gengy_init_values"):
        children = [(typ[1], i.gengy_init_values[idx]) for idx, typ in enumerate(get_arguments(i))]
    else:
        assert False, f"Unsupported: {i} ({type(i)})"

    for t, c in children:
        nodes, dist, thisway, weighted_nodes = relabel_nodes(
            c,
            g,
            isinstance(c, list),
        )
        abs_adjust = 0 if not is_abstract(t) or not g.expansion_depthing else g.abstract_dist_to_t[t][type(c)]
        if isinstance(c, list) and g.expansion_depthing:
            abs_adjust = 1
        list_adjust = 0 if isinstance(c, list) else 1
        number_of_nodes += abs_adjust + nodes
        weighted_number_of_nodes += weighted_nodes
        distance_to_term = max(distance_to_term, dist + abs_adjust + list_adjust)
        for k, v in thisway.items():
            types_this_way[k].extend(v)

    if not is_list:
        weighted_number_of_nodes += distance_to_term

    i.gengy_labeled = True
    i.gengy_distance_to_term = distance_to_term
    i.gengy_nodes = number_of_nodes
    i.gengy_weighted_nodes = weighted_number_of_nodes
    i.gengy_types_this_way = types_this_way
    return number_of_nodes, distance_to_term, types_this_way, weighted_number_of_nodes


def relabel_nodes_of_trees(i: TreeNode, g: Grammar) -> TreeNode:
    """Recomputes all the nodes, depth and distance_to_term in the tree."""

    relabel_nodes(i, g)
    return i


T = TypeVar("T")


def tree_node_fold(i: TreeNode, f: Callable[[Any, list[T]], T]):
    """Recursively folds over all elements of the tree."""
    ty = type(i)
    if isinstance(i, list):
        return f(i, [tree_node_fold(n, f) for n in i])
    elif dataclasses.is_dataclass(i):
        return f(i, [tree_node_fold(getattr(i, aname), f) for (aname, _) in get_arguments(ty)])
    else:
        return f(i, [])
