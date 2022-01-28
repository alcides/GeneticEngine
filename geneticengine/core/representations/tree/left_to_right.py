from dataclasses import dataclass
from typing import Any, Callable, Optional, Type, Dict, List

from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.tree.utils import (
    relabel_nodes_of_trees,
    get_depth,
)
from geneticengine.core.representations.tree.wrapper import Wrapper
from geneticengine.core.utils import get_arguments

def create_left_to_right_grow(
    expand_node: Any,  # Callable[[Source, Grammar, int, Type, str, Dict[str, Type]], Any]
):
    flag = False
    
    def left_to_right_grow(
        r: Source,
        g: Grammar,
        max_depth: int,
        starting_symbol: Type[Any] = int,
        force_depth:bool = False,
    ):
        nonlocal flag
        root = expand_node(
            r, g, max_depth, starting_symbol, "", {}, has_reached_final_depth=flag
        )
        for (name, ty) in get_arguments(root):
            v = getattr(root, name)
            if isinstance(v, Wrapper):
                setattr(root, name, left_to_right_grow(r, g, max_depth-1, ty, force_depth)) 
        return root

    return left_to_right_grow
