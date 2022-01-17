from dataclasses import dataclass
from typing import Any, Callable, Type, Dict, List

from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import Source
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
        depth: int,
        starting_symbol: Type[Any] = int,
    ):
        root = expand_node(r, g, depth, starting_symbol, "", {})
        prod_queue: List[Future] = [root]

        while prod_queue:
            index = r.randint(0, len(prod_queue) - 1)
            future = prod_queue.pop(index)
            if isinstance(future, Future):
                obj = expand_node(
                    r,
                    g,
                    depth=future.depth,
                    starting_symbol=future.ty,
                    argname=future.name,
                    context=future.context,
                )
                setattr(future.parent, future.name, obj)
            else:
                obj = future  # only for root

            prod_queue.extend(extract_futures(obj))

        assert isinstance(root, starting_symbol)
        return root

    return position_independent_grow
