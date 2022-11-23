from __future__ import annotations

from abc import ABC
from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import make_dataclass
from math import isnan
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


# type (object, bases, dict)
def create_dataclass_dynamically(
    name: str,
    args: dict[str, Any] = {},
    annotations: dict[str, Any] = {},
    parent_class: object = ABC,
) -> type:
    def init(self, *x, **y):
        pass

    args["__init__"] = init
    new_data_class = type(name, (parent_class,), args)

    if annotations:
        assert hasattr(new_data_class.__init__, "__annotations__")
        for key in annotations:
            new_data_class.__init__.__annotations__[key] = annotations[key]

    return new_data_class


def create_grammar_nodes(
    seed: int,
    n_class_abc: int,
    n_class_0_children: int,
    n_class_2_children: int,
    min_depth: int = 0,
    recursion_p=0,
) -> list[type]:
    # how to choose the parent class??? randomly???
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

    nodes = abc_classes + children0_classes + children2_classes

    return nodes


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
            annotation = {}

            if terminals:
                rand_idx_terminals = random_source.randint(0, len(terminals) - 1)
                random_terminal = terminals[rand_idx_terminals]
                annotation["x"] = random_terminal
            else:
                annotation["x"] = int

            node = create_dataclass_dynamically(
                name=name + str(i),
                parent_class=random_parent,
                annotations=annotation,
            )

        return_list.append(node)

    return return_list


def create_grammar_dynamically(
    seed: int,
    class_abc: list[object],
    class_0_children: list[object],
    class_2_children: list[object],
    min_depth: int,
    recursion_p,
) -> Grammar:

    random_source = RandomSource(seed)
    rand_idx_abc = random_source.randint(0, len(class_abc))
    starting_symbol = class_abc[rand_idx_abc]

    if isinstance(starting_symbol, tuple):
        starting_symbol = create_dataclass_dynamically(
            starting_symbol[0],
            starting_symbol[1],
            starting_symbol[2],
            starting_symbol[3],
        )

    children_classes = class_0_children + class_2_children

    children_classes.append(c for c in class_abc if c is not starting_symbol)

    considered_subtypes = []
    for c in children_classes:
        if isinstance(c, type):
            considered_subtypes.append(c)
        elif isinstance(c, tuple):
            c_aux = create_dataclass_dynamically(c[0], c[1], c[2], c[3])
            considered_subtypes.append(c_aux)

    assert isinstance(starting_symbol, type)
    assert isinstance(children_classes, list)
    return extract_grammar(
        children_classes,
        starting_symbol,
    )
