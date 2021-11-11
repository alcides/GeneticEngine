import sys
from copy import deepcopy

from typing import (
    Any,
    Callable,
    Type,
    TypeVar,
    Tuple,
    List,
    Union,
)

from geneticengine.core.random.sources import RandomSource, Source
from geneticengine.core.grammar import Grammar
from geneticengine.core.representations.base import Representation
from geneticengine.core.tree import TreeNode
from geneticengine.core.utils import (
    get_arguments,
    is_generic_list,
    get_generic_parameter,
    is_terminal,
)
from geneticengine.exceptions import GeneticEngineError


def random_int(r: Source) -> int:
    return r.randint(-(sys.maxsize - 1), sys.maxsize)


def random_float(r: Source) -> float:
    return r.random_float(-100, 100)


T = TypeVar("T")


def random_list(
    r: Source, rec: Callable[[Type[Any]], Any], depth: int, ty: Type[List[T]]
) -> List[T]:
    inner_type = get_generic_parameter(ty)
    size = r.randint(0, depth)
    return [rec(inner_type) for _ in range(size)]


def is_metahandler(ty: type) -> bool:
    """
    Returns if type is a metahandler.
    AnnotatedType[int, IntRange(3,10)] is an example of a Metahandler.

    Verification is done using the __metadata__, which is the first argument of Annotated
    """
    return hasattr(ty, "__metadata__")


def apply_metahandler(
    r: Source,
    rec: Callable[[Type[Any]], Any],
    ty: Type[Any],
) -> Any:
    """
    This method applies a metahandler to use a custom generator for things of a given type.

    As an example, AnnotatedType[int, IntRange(3,10)] will use the IntRange.generate(r, recursive_generator)

    The generator is the annotation on the type ("__metadata__").
    """
    metahandler = ty.__metadata__[0]
    base_type = get_generic_parameter(ty)
    return metahandler.generate(r, lambda: rec(base_type))


def random_node(
    r: Source,
    g: Grammar,
    depth: int = 5,
    starting_symbol: Type[Any] = int,
) -> Any:
    """
    Creates a random node of a given type (starting_symbol)
    """
    if depth < 0:
        raise GeneticEngineError("Recursion Depth reached")
    if depth < g.get_distance_to_terminal(starting_symbol):
        raise GeneticEngineError(
            "There will be no depth sufficient for symbol {} in this grammar (provided: {}, required: {}).".format(
                starting_symbol, depth, g.get_distance_to_terminal(starting_symbol)
            )
        )

    recursive_generator = lambda t: random_node(r, g, depth - 1, t)

    if starting_symbol is int:
        return random_int(r)
    elif starting_symbol is float:
        return random_float(r)
    elif is_generic_list(starting_symbol):
        return random_list(r, recursive_generator, depth, starting_symbol)
    elif is_metahandler(starting_symbol):
        node = apply_metahandler(r, recursive_generator, starting_symbol)
        return relabel_nodes_of_trees(node, g.non_terminals())
    else:
        if starting_symbol not in g.productions:
            raise GeneticEngineError(f"Symbol {starting_symbol} not in grammar rules.")

        compatible_productions = g.productions[starting_symbol]
        valid_productions = [
            vp for vp in compatible_productions if g.distanceToTerminal[vp] <= depth
        ]
        if not valid_productions:
            raise GeneticEngineError(
                "No productions for non-terminal node with type: {} in depth {} (minimum required: {}).".format(
                    starting_symbol,
                    depth,
                    str(
                        [
                            (vp, g.distanceToTerminal[vp])
                            for vp in compatible_productions
                        ]
                    ),
                )
            )
        rule = r.choice(valid_productions)
        args = [recursive_generator(at) for (a, at) in get_arguments(rule)]
        node = rule(*args)
        node = relabel_nodes_of_trees(node, g.non_terminals())
        return node


def random_individual(r: Source, g: Grammar, max_depth: int = 5) -> TreeNode:
    assert max_depth >= g.get_min_tree_depth()
    ind = random_node(r, g, max_depth, g.starting_symbol)
    assert isinstance(ind, TreeNode)
    return ind


def mutate_inner(r: Source, g: Grammar, i: TreeNode, max_depth: int) -> TreeNode:
    if i.nodes > 0:
        c = r.randint(0, i.nodes - 1)
        if c == 0:
            ty = i.__class__.__bases__[0]
            try:
                replacement = random_node(r, g, max_depth - i.depth + 1, ty)
                return replacement
            except:
                return i
        else:
            for field in i.__annotations__:
                child = getattr(i, field)
                if hasattr(child, "nodes"):
                    count = child.nodes
                    if c <= count:
                        setattr(i, field, mutate_inner(r, g, child, max_depth))
                        return i
                    else:
                        c -= count
            return i
    else:
        return i


def mutate(r: Source, g: Grammar, i: TreeNode, max_depth: int) -> Any:
    new_tree = mutate_inner(r, g, deepcopy(i), max_depth)
    relabeled_new_tree = relabel_nodes_of_trees(new_tree, g.non_terminals())
    return relabeled_new_tree


def find_in_tree(ty: type, o: TreeNode, max_depth: int):
    if ty in o.__class__.__bases__ and o.distance_to_term <= max_depth:
        yield o
    if hasattr(o, "__annotations__"):
        for field in o.__annotations__:
            child = getattr(o, field)
            yield from find_in_tree(ty, child, max_depth)


def tree_crossover_inner(
    r: Source, g: Grammar, i: TreeNode, o: TreeNode, max_depth: int
) -> Any:
    if i.nodes > 0:
        c = r.randint(0, i.nodes - 1)
        if c == 0:
            ty = i.__class__.__bases__[0]
            replacement = None
            options = list(find_in_tree(ty, o, max_depth - i.depth + 1))
            if options:
                replacement = r.choice(options)
            if replacement is None:
                try:
                    replacement = random_node(r, g, max_depth - i.depth + 1, ty)
                except:
                    return i
            return replacement
        else:
            for field in i.__annotations__:
                child = getattr(i, field)
                if hasattr(child, "nodes"):
                    count = getattr(i, field).nodes
                    if c <= count:
                        setattr(
                            i,
                            field,
                            tree_crossover_inner(r, g, getattr(i, field), o, max_depth),
                        )
                        return i
                    else:
                        c -= count
            return i
    else:
        return i


def tree_crossover(
    r: Source, g: Grammar, p1: TreeNode, p2: TreeNode, max_depth: int
) -> Tuple[TreeNode, TreeNode]:
    """
    Given the two input trees [p1] and [p2], the grammar and the random source, this function returns two trees that are created by crossing over [p1] and [p2]. The first tree returned has [p1] as the base, and the second tree has [p2] as a base.
    """
    new_tree1 = tree_crossover_inner(r, g, deepcopy(p1), deepcopy(p2), max_depth)
    relabeled_new_tree1 = relabel_nodes_of_trees(new_tree1, g.non_terminals())
    new_tree2 = tree_crossover_inner(r, g, deepcopy(p2), deepcopy(p1), max_depth)
    relabeled_new_tree2 = relabel_nodes_of_trees(new_tree2, g.non_terminals())
    return relabeled_new_tree1, relabeled_new_tree2


def tree_crossover_single_tree(
    r: Source, g: Grammar, p1: TreeNode, p2: TreeNode, max_depth: int
) -> TreeNode:
    """
    Given the two input trees [p1] and [p2], the grammar and the random source, this function returns one tree that is created by crossing over [p1] and [p2]. The tree returned has [p1] as the base.
    """
    new_tree = tree_crossover_inner(r, g, deepcopy(p1), deepcopy(p2), max_depth)
    relabeled_new_tree = relabel_nodes_of_trees(new_tree, g.non_terminals())
    return relabeled_new_tree


def get_property_names(obj: TreeNode) -> List[Any]:
    if hasattr(obj, "__annotations__"):
        return [field for field in obj.__annotations__]
    else:
        return []


def relabel_nodes_of_trees(
    i: TreeNode, non_terminals: list[type], max_depth: int = 1
) -> TreeNode:
    """ Recomputes all the nodes, depth and distance_to_term in the tree """
    # print("Node: {}, nodes: {}, distance_to_term: {}, depth: {}.".format(i,i.nodes,i.distance_to_term,i.depth))
    def relabel_nodes(i: TreeNode, depth: int = 1) -> Tuple[int, int]:
        if is_terminal(type(i), non_terminals):
            if type(i) in non_terminals:
                i.depth = depth
                i.distance_to_term = 1
                i.nodes = 0
            return (0, 1)
        elif isinstance(i, list):
            children = i
        else:
            children = [getattr(i, field) for field in get_property_names(i)]
        assert children
        properties_of_children = [relabel_nodes(child, depth + 1) for child in children]
        number_of_nodes = 1 + sum([prop[0] for prop in properties_of_children])
        distance_to_term = 1 + max([prop[1] for prop in properties_of_children])
        if not isinstance(i, list):
            i.depth = depth
            i.distance_to_term = distance_to_term
            i.nodes = number_of_nodes
        return number_of_nodes, distance_to_term

    # print("Node: {}, nodes: {}, distance_to_term: {}, depth: {}.".format(i,i.nodes,i.distance_to_term,i.depth))
    relabel_nodes(i, max_depth)
    return i


treebased_representation = Representation(
    create_individual=random_individual,
    mutate_individual=mutate,
    crossover_individuals=tree_crossover,
    genotype_to_phenotype=lambda g, x: x,
)
