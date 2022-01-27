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


@dataclass
class Future(object):
    parent: Any
    name: str
    depth: int
    ty: Type
    context: Dict[str, Type]


def extract_futures(obj: Any) -> List[Future]:
    futures = []
    if isinstance(obj, list):
        for el in obj:
            if isinstance(el, Future):
                futures.append(el)
            else:
                futures.extend(extract_futures(el))
    else:
        for (name, ty) in get_arguments(obj):
            v = getattr(obj, name)
            if isinstance(v, Wrapper):
                context = {argn: argt for (argn, argt) in get_arguments(obj)}
                futures.append(Future(obj, name, v.depth, v.target, context))
            if isinstance(v, Future):
                futures.append(v)
            # Do we want a full recursion here? probably not for performance reasons
            # else:
            #    futures.extend(extract_futures(v))
    return futures


def create_position_independent_grow(
    expand_node: Any,  # Callable[[Source, Grammar, int, Type, str, Dict[str, Type]], Any]
):
    def position_independent_grow(
        r: Source,
        g: Grammar,
        max_depth: int,
        starting_symbol: Type[Any] = int,
        force_depth: Optional[int] = None,
    ):
        has_reached_final_depth = force_depth is None
        root = expand_node(
            r, g, max_depth, starting_symbol, "", {}, has_reached_final_depth=False
        )
        prod_queue: List[Future] = [root]

        while prod_queue:
            index = r.randint(0, len(prod_queue) - 1)
            future = prod_queue.pop(index)
            if isinstance(future, Future):
                expected_depth = future.depth
                obj = expand_node(
                    r,
                    g,
                    max_depth=future.depth,
                    starting_symbol=future.ty,
                    argname=future.name,
                    context=future.context,
                    has_reached_final_depth=has_reached_final_depth,
                )
                setattr(future.parent, future.name, obj)
            else:
                expected_depth = force_depth
                obj = future  # only for root
            new_futures = extract_futures(obj)

            real_depth = get_depth(g, obj)
            if (
                not has_reached_final_depth
                and not new_futures
                and real_depth == expected_depth
            ):
                has_reached_final_depth = True
            prod_queue.extend(new_futures)
        assert isinstance(root, starting_symbol)
        return root

    return position_independent_grow
