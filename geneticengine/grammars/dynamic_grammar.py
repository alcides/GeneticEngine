from __future__ import annotations

from abc import ABC
from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import make_dataclass
from math import isnan
import string
from typing import Annotated
from typing import Any
from typing import Callable

import numpy as np

from geneticengine.core.decorators import abstract
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.tree import TreeNode
from geneticengine.grammars.sgp import Number
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.vars import VarRange

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
    
    # TODO making sure that the generated classes have the same name length 
    nodes = []
    abc_classes = create_nodes_list_aux(seed, "abc_", n_class_abc)

    children0_classes = create_nodes_list_aux(
        seed,
        "terminal_",
        n_class_0_children,
        parent_list=abc_classes,
    )

    children2_classes = create_nodes_list_aux(
        seed,
        "non_terminal_",
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
    size: int,
    parent_list: list = [],
    terminals: list = [],
) -> list[type]:
    return_list = []
    random_source = RandomSource(seed)
    
    for i in range(size):
        
        if not parent_list:
            node = abstract(create_dataclass_dynamically(name + str(i)))
        else:
            rand_idx_abc = random_source.randint(0, len(parent_list) - 1)
            random_parent = parent_list[rand_idx_abc]
            
            annotations_aux = create_random_annotations(
                random_source, terminals)

            node = create_dataclass_dynamically(
                name=name + str(i),
                parent_class=random_parent,
                annotations=annotations_aux,
            )

        return_list.append(node)

    return return_list


def create_random_annotations (
    random_source : RandomSource,
    terminals: list ,
)-> dict[str, type]:
    annotations= {}
    var_letters = list(string.ascii_lowercase)
    
    #I tried it with 10 random annotations, but the grammar was getting too long
    for i in range(random_source.randint(1, 6)):
        if terminals:
            rand_idx_terminals = random_source.randint(0, len(terminals) - 1)
            random_terminal = terminals[rand_idx_terminals]
            annotations[var_letters[i]] = random_terminal
            
        else:
            rand_idx_types = random_source.randint(0, len(primitive_types) - 1)
            random_type = primitive_types[rand_idx_types]
            annotations[var_letters[i]] = random_type
            
    return annotations


def edit_distance(string1, string2):

    if len(string1) > len(string2):
        difference = len(string1) - len(string2)
        string1[:difference]

    elif len(string2) > len(string1):
        difference = len(string2) - len(string1)
        string2[:difference]

    else:
        difference = 0

    for i in range(len(string1)):
        if string1[i] != string2[i]:
            difference += 1

    return difference
