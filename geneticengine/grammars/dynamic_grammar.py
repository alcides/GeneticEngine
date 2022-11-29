from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
import string

from geneticengine.core.decorators import abstract
from geneticengine.core.random.sources import RandomSource


#TODO: add more types for the grammar terminals
primitive_types= [int, float]

# type (object, bases, dict)
def create_dataclass_dynamically(
    name: str,
    args: dict[str, Any] = {},
    annotations: dict[str, Any] = {},
    parent_class: object = ABC,
) -> type:
    new_data_class = type(name, (parent_class,), args)
    
    if annotations:
        for key in annotations:
            new_data_class.__annotations__[key] = annotations[key]

    return dataclass(new_data_class)


def create_grammar_nodes(
    seed: int,
    n_class_abc: int,
    n_class_0_children: int,
    n_class_2_children: int,
    min_depth: int = 0,
    recursion_p=0,
) -> tuple[list[type], type]:
    
    max_digit = max([str(n_class_abc), str(n_class_0_children), str(n_class_2_children)], key= len)
    max_class_name_length= 10 + int(max_digit)
    
    nodes = []                                 
    abc_classes = create_nodes_list_aux(seed, "class_abc_", max_class_name_length, n_class_abc)

    children0_classes = create_nodes_list_aux(
        seed,
        "tterminal_",
        max_class_name_length,
        n_class_0_children,
        parent_list=abc_classes,
    )

    children2_classes = create_nodes_list_aux(
        seed,
        "nterminal_",
        max_class_name_length,
        n_class_2_children,
        parent_list=abc_classes,
        terminals=children0_classes,
    )

    nodes =  children0_classes + children2_classes
    
    random_source = RandomSource(seed)
    rand_idx_abc = random_source.randint(0, len(abc_classes) - 1)
    random_starting_node = abc_classes[rand_idx_abc]
    
    return (nodes, random_starting_node)


def create_nodes_list_aux(
    seed: int,
    name: str,
    name_length: int,
    size: int,
    parent_list: list = [],
    terminals: list = [],
) -> list[type]:
    return_list = []
    random_source = RandomSource(seed)
    
    for i in range(size):
        
        name_class = (name + str(i)).ljust(name_length, '0')
        
        if not parent_list:
            node = abstract(create_dataclass_dynamically(name_class))
        else:
            rand_idx_abc = random_source.randint(0, len(parent_list) - 1)
            random_parent = parent_list[rand_idx_abc]
            
            annotations_aux = create_random_annotations(
                random_source, terminals, 10)
            
            node = create_dataclass_dynamically(
                name=name_class,
                parent_class=random_parent,
                annotations=annotations_aux,
            )

        return_list.append(node)

    return return_list


def create_random_annotations (
    random_source : RandomSource,
    terminals: list ,
    n_annotations:int,
)-> dict[str, type]:
    annotations= {}
    var_letters = list(string.ascii_lowercase)
    
    for i in range(random_source.randint(1, n_annotations)):
        if terminals:
            rand_idx_terminals = random_source.randint(0, len(terminals) - 1)
            random_terminal = terminals[rand_idx_terminals]
            annotations[var_letters[i]] = random_terminal
            
        else:
            rand_idx_types = random_source.randint(0, len(primitive_types) - 1)
            random_type = primitive_types[rand_idx_types]
            annotations[var_letters[i]] = random_type
            
    return annotations


def edit_distance(string1: str, string2: str) -> int:

    if len(string1) > len(string2):
        string1, string2 = string2, string1

    distances = range(len(string1) + 1)
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
