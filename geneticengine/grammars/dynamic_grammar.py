from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
import string
from typing import Any

from geneticengine.core.decorators import abstract
from geneticengine.core.random.sources import RandomSource

#TODO: add more types for the grammar terminals
primitive_types= [int, float]

def create_grammar_nodes(
    seed: int,
    n_class_abc: int,
    n_class_0_children: int,
    n_class_2_children: int,
    max_var_per_class: int = 5,
) -> tuple[list[type], type]:
    """
    Creates a list of data classes and a starting data class for a grammar.

    Args:
        seed (int): the seed for the random number generator.
        n_class_abc (int): the number of abstract base classes to create.
        n_class_0_children (int): the number of terminal classes to create.
        n_class_2_children (int): the number of non-terminal classes to create.
        max_var_per_class (int): the maximum number of variables that each data class can have. Default is 5.

    Returns:
        tuple[list[type], type]: a tuple containing a list of the new data classes and the starting data class for a grammar.
    """
    random_source = RandomSource(seed)
    
    max_digit = max([str(n_class_abc), str(n_class_0_children), str(n_class_2_children)], key= len)
    max_class_name_length= 10 + int(max_digit)
    
    nodes = []                                 
    abc_classes = create_nodes_list_aux(
        random_source, "class_abc_", max_class_name_length, n_class_abc)

    children0_classes = create_nodes_list_aux(
        random_source,
        "tterminal_",
        max_class_name_length,
        n_class_0_children,
        max_vars=max_var_per_class,
        parent_list=abc_classes,
    )

    children2_classes = create_nodes_list_aux(
        random_source,
        "nterminal_",
        max_class_name_length,
        n_class_2_children,
        max_vars= max_var_per_class,
        parent_list=abc_classes,
        terminals=children0_classes,
    )

    nodes =  children0_classes + children2_classes
    
    random_starting_node = random_node_from_list(random_source, abc_classes)
    
    return (nodes, random_starting_node)


# type (object, bases, dict)
def create_dataclass_dynamically(
    name: str,
    args: dict[str, Any] = {},
    annotations: dict[str, Any] = {},
    parent_class: type = ABC,
) -> type:
    """
    Dynamically creates a new data class with the given name and arguments.

    Args:
        name (str): the name of the new data class
        args (dict[str, Any]): the attributes and values for the new data class. Default is an empty dictionary.
        annotations (dict[str, Any]): the type annotations for the new data class. Default is an empty dictionary.
        parent_class (type): the parent class for the new data class. Default is ABC (abstract base class).

    Returns:
        The new data class created
    """
    new_data_class = type(name, (parent_class,), args)

    if annotations:
        for key in annotations:
            new_data_class.__annotations__[key] = annotations[key]

    return dataclass(new_data_class)


def create_nodes_list_aux(
    random_source: RandomSource,
    name: str,
    name_length: int,
    size: int,
    max_vars: int= 5,
    parent_list: list = [],
    terminals: list = [],
    
) -> list[type]:
    """
    Creates a list of data classes with the given size, max variables, parent classes and annotations.

    Args:
        random_source (RandomSource):  an instance of the `RandomSource` class, which is used to generate random values.
        name (str): the base name for the new data classes.
        name_length (int): the desired length of the names of the new data classes.
        size (int): the number of data classes to be created.
        max_vars (int): the maximum number of variables that each data class can have. Default is 5.
        parent_list (list): the list of parent classes for the new data classes. Default is an empty list.
        terminals (list): the list of terminal symbols for the new data classes. Default is an empty list.

    Returns:
        a list of the new data classes.
    """
    return_list = []
    
    for i in range(size):
        
        name_class = (name + str(i)).ljust(name_length, 'x')
        
        if not parent_list:
            node = abstract(create_dataclass_dynamically(name_class))
        else: 
            random_parent= random_node_from_list(random_source, parent_list)
            
            random_annotations = create_random_annotations(
                random_source, terminals, max_vars)
            
            node = create_dataclass_dynamically(
                name=name_class,
                parent_class=random_parent,
                annotations=random_annotations,
            )

        return_list.append(node)

    return return_list


def create_random_annotations (
    random_source : RandomSource,
    terminals: list ,
    n_annotations:int,
)-> dict[str, type]:
    """
    Create a dictionary of random annotations.
    
    Args:
        random_source (RandomSource): an instance of the `RandomSource` class, which is used to generate a random integer.
        terminals (list): a list of terminal nodes from which to select random nodes.
        n_annotations (int): the maximum number of annotations to create.
    
    Returns:
        A dictionary of random annotations, where the keys are strings (representing variable names) and the values are objects of some unspecified type (representing the type of the variable).
    """
    
    annotations = {}
    var_letters = list(string.ascii_lowercase)
    
    for i in range(random_source.randint(1, n_annotations)):
        random_terminal = random_node_from_list(random_source, terminals if terminals else primitive_types)
        annotations[var_letters[i]] = random_terminal
            
    return annotations


def random_node_from_list(random_source: RandomSource, node_list: list) -> type:
    """
    Return a random node from a list of nodes.
    
    Args:
        random_source (RandomSource): an instance of the `RandomSource` class, which is used to generate a random integer.
        node_list (list): a list of nodes from which to select a random node.
    
    Returns:
        A random node from the list of nodes.
    """
    rand_idx = random_source.randint(0, len(node_list) - 1)
    return node_list[rand_idx]


def edit_distance(string1: str, string2: str) -> int:
    """
    The edit distance is the minimum number of operations (insertions, deletions, or substitutions)
    needed to transform one string into another.
    
    Args:
        string1 (str) - The first string to compare.
        string2 (str) - The second string to compare.
        
    Returns:
        The edit distance (int) between `string1` and `string2`.
    """

    if len(string1) > len(string2):
        string1, string2 = string2, string1

    distances = list(range(len(string1) + 1))
    for i2, c2 in enumerate(string2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(string1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(
                    1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
        
    return distances[-1]
