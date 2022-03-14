from collections import defaultdict
from functools import reduce
from typing import Any, List, Set, Tuple, Dict
from geneticengine.core.grammar import Grammar
from geneticengine.core.tree import TreeNode
from geneticengine.core.utils import is_terminal
from geneticengine.core.decorators import is_builtin


def get_property_names(obj: TreeNode) -> List[Any]:
    if hasattr(obj, "__annotations__"):
        return [field for field in obj.__annotations__]
    else:
        return []


def get_depth(g: Grammar, i: Any) -> int:
    if is_builtin(type(i)):
        return 1
    elif is_terminal(type(i), g.non_terminals):
        return 1
    elif isinstance(i, list):
        children = i
    else:
        children = [getattr(i, field) for field in get_property_names(i)]
    return 1 + max([get_depth(g, i) for i in children])


# print("Node: {}, nodes: {}, distance_to_term: {}, depth: {}.".format(i,i.nodes,i.distance_to_term,i.depth))
def relabel_nodes(
    i: TreeNode, non_terminals: Set[type]
) -> Tuple[int, int, Dict[type, List[Any]]]:
    children: List[Any]
    if getattr(i, "gengy_labeled", False):
        return i.gengy_nodes, i.gengy_distance_to_term, i.gengy_types_this_way
    if is_terminal(type(i), non_terminals):
        if not is_builtin(type(i)):
            i.gengy_labeled = True
            i.gengy_distance_to_term = 1
            i.gengy_nodes = 0
            i.gengy_types_this_way = {type(i): [i]}
        return 0, 1, {type(i): [i]}
    elif isinstance(i, list):
        children = i
    else:
        children = [getattr(i, field) for field in get_property_names(i)]
    assert children
    properties_of_children = [relabel_nodes(child, non_terminals) for child in children]
    number_of_nodes = 1 + sum([prop[0] for prop in properties_of_children])
    distance_to_term = 1 + max([prop[1] for prop in properties_of_children])
    # print(dir(type(i).__init__.__code__.co_filename))
    types_this_way = defaultdict(lambda: [])
    types_this_way[type(i)] = [i]
    for (_, _, types) in properties_of_children:
        types_this_way.update(types)
    if not isinstance(i, list):
        i.gengy_labeled = True
        i.gengy_distance_to_term = distance_to_term
        i.gengy_nodes = number_of_nodes
        i.gengy_types_this_way = types_this_way
    return number_of_nodes, distance_to_term, types_this_way


def relabel_nodes_of_trees(i: TreeNode, non_terminals: Set[type]) -> TreeNode:
    """Recomputes all the nodes, depth and distance_to_term in the tree"""

    # print("Node: {}, nodes: {}, distance_to_term: {}, depth: {}.".format(i,i.nodes,i.distance_to_term,i.depth))
    relabel_nodes(i, non_terminals)
    return i
