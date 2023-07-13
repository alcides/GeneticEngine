from __future__ import annotations

from collections import defaultdict
from typing import Any, Generic, TypeVar

from geneticengine.core.decorators import is_builtin
from geneticengine.core.grammar import Grammar
from geneticengine.core.tree import TreeNode
from geneticengine.core.utils import get_arguments
from geneticengine.core.utils import is_abstract
from geneticengine.core.utils import is_terminal


T = TypeVar("T")


class GengyList(list, Generic[T]):
    def __init__(self, typ, vals):
        super().__init__(vals)
        self.typ = typ
        self.gengy_init_values = vals

    def new_like(self, *newargs):
        return GengyList(self.typ, newargs)

    def __hash__(self):
        return sum(hash(o) for o in self)


def relabel_nodes(
    i: TreeNode,
    g: Grammar,
    is_list: bool = False,
) -> tuple[int, int, dict[type, list[Any]], int]:
    """Recomputes node specifics in the tree.\n
    Returns the number of nodes, distance to terminal (depth), typed this way,
    and the weighted number of nodes (counting depth points instead of
    nodes)."""
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
    if is_terminal(type(i), non_terminals) and (not isinstance(i, list)):
        if not is_builtin(type(i)):
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
    else:
        if isinstance(i, list):
            children = [(type(obj), obj) for obj in i]
        else:
            if not hasattr(i, "gengy_init_values"):
                breakpoint()
            children = [(typ[1], i.gengy_init_values[idx]) for idx, typ in enumerate(get_arguments(i))]
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

def get_nodes_depth_specific(i: TreeNode, g: Grammar):
    depth = i.gengy_distance_to_term
    if not i.gengy_labeled:
        relabel_nodes_of_trees(i, g)
    
    nodes_depth_specific: dict[str, float] = dict()
    def add_count(node: TreeNode, n_d_spec_dict: dict[str, float]):
        if hasattr(node, "gengy_distance_to_term"):
            try: 
                n_d_spec_dict[str(depth - node.gengy_distance_to_term)] += 1
            except Exception:
                n_d_spec_dict[str(depth - node.gengy_distance_to_term)] = 1

        if not (is_terminal(type(node), g.non_terminals) and (not isinstance(node, list))):
            if isinstance(node, list):
                children = [(type(obj), obj) for obj in node]
            else:
                if not hasattr(node, "gengy_init_values"):
                    breakpoint()
                children = [(typ[1], node.gengy_init_values[idx]) for idx, typ in enumerate(get_arguments(node))]
            for t, c in children:
                n_d_spec_dict = add_count(c, n_d_spec_dict)
        return n_d_spec_dict
    
    add_count(i, nodes_depth_specific)
    return nodes_depth_specific
