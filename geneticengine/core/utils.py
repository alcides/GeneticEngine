from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from functools import wraps
from typing import Any
from typing import Callable
from typing import get_type_hints
from typing import List
from typing import Protocol
from typing import Set
from typing import Tuple
from typing import Type

from geneticengine.core.decorators import get_gengy


def is_annotated(ty: type[Any]):
    """Returns whether type is annotated with metadata."""
    return hasattr(ty, "__metadata__")


def is_generic_list(ty: type[Any]):
    """Returns whether a type is List[T] for any T"""
    return hasattr(ty, "__origin__") and ty.__origin__ is list


def is_generic(ty: type[Any]):
    """Returns whether a type is x[T] for any T"""
    return hasattr(ty, "__origin__")


def get_generic_parameters(ty: type[Any]) -> list[type]:
    """Annotated[T, <annotations>] or List[T], this function returns Dict[T,]"""
    return ty.__args__


def get_generic_parameter(ty: type[Any]) -> type:
    """When given Annotated[T, <annotations>] or List[T], this function returns T"""
    return get_generic_parameters(ty)[0]


def strip_annotations(ty: type[Any]) -> type:
    """When given Annotated[T, <annotations>] or List[T], this function recurses with T
    Otherwise, it returns the parameter unchanged.
    """
    if is_generic_list(ty) or is_annotated(ty):
        return strip_annotations(get_generic_parameter(ty))
    else:
        return ty


def has_arguments(n: Any) -> bool:
    """Returns whether a node has arguments or not"""
    return (
        hasattr(n, "__init__")
        and hasattr(n.__init__, "__annotations__")
        and len(n.__init__.__annotations__) > 0
    )


def get_arguments(n) -> list[tuple[str, type]]:
    """
    :param n: production
    :return: list((argname, argtype))
    """
    if hasattr(n, "__init__"):
        init = n.__init__
        import sys

        args: dict[str, type] = get_type_hints(
            init,
            globalns=sys.modules[n.__module__].__dict__,
            include_extras=True,
        )
        return [(a, args[a]) for a in filter(lambda x: x != "return", args)]
    return []


def is_abstract(t: type) -> bool:
    """Returns whether a class is a Protocol or AbstractBaseClass"""
    return t.mro()[1] in [ABC, Protocol] or get_gengy(t).get("abstract", False)


def is_terminal(t: type, l: set[type]) -> bool:
    """Returns whether a node is a terminal or not, based on the list of non terminals in the grammar"""
    if is_annotated(t):
        return all([is_terminal(inner, l) for inner in get_generic_parameters(t)])
    if not has_arguments(t):
        return True
    return t not in l


# debug_fin = [0]


def build_finalizers(
    final_callback,
    n_args,
    per_callback: list[Callable[[Any], None]] = None,
) -> list[Any]:
    """
    Builds a set of functions that accumulate the arguments provided
    :param final_callback:
    :param n_args:
    :return:
    """
    uninit = object()
    rets = [uninit] * n_args
    to_arrive = [n_args]

    finalizers = []
    # id = debug_fin[0]
    # print("%i has %i fin " % (id, n_args))
    # debug_fin[0] += 1

    for i in range(n_args):

        def fin(x, i=i):
            if rets[i] is uninit:
                rets[i] = x
                to_arrive[0] -= 1

                if per_callback is not None:
                    per_callback[i](x)

                if to_arrive[0] == 0:
                    # we recieved all params, finish construction
                    final_callback(*rets)
                # else:
                #     print("%i prog %i" % (id, to_arrive[0]))
            else:
                raise Exception(
                    "Received duplicate param on finalizer! i=%d" % i,
                )

        finalizers.append(fin)

    if n_args == 0:
        final_callback()

    return finalizers
