import logging
import sys
from abc import ABC
from typing import (
    Any,
    Protocol,
    Set,
    Type,
    Tuple,
    List,
    Callable,
)

from geneticengine.core.decorators import get_gengy


def is_annotated(ty: Type[Any]):
    """Returns whether type is annotated with metadata."""
    return hasattr(ty, "__metadata__")


def is_generic_list(ty: Type[Any]):
    """Returns whether a type is List[T] for any T"""
    return hasattr(ty, "__origin__") and ty.__origin__ is list


def get_generic_parameters(ty: Type[Any]) -> list[type]:
    """Annotated[T, <annotations>] or List[T], this function returns Dict[T,]"""
    return ty.__args__


def get_generic_parameter(ty: Type[Any]) -> type:
    """When given Annotated[T, <annotations>] or List[T], this function returns T"""
    return get_generic_parameters(ty)[0]


def strip_annotations(ty: Type[Any]) -> type:
    """When given Annotated[T, <annotations>] or List[T], this function recurses with T
    Otherwise, it returns the parameter unchanged.
    """
    if is_generic_list(ty) or is_annotated(ty):
        return strip_annotations(get_generic_parameter(ty))
    else:
        return ty


def get_arguments(n) -> List[Tuple[str, type]]:
    """
    :param n: production
    :return: list((argname, argtype))
    """
    if hasattr(n, "__annotations__"):
        args = n.__annotations__
        return [(a, args[a]) for a in args]
    else:
        return []


def is_abstract(t: type) -> bool:
    """Returns whether a class is a Protocol or AbstractBaseClass"""
    return t.mro()[1] in [ABC, Protocol] or get_gengy(t).get("abstract", False)


def is_terminal(t: type, l: Set[type]) -> bool:
    """Returns whether a node is a terminal or not, based on the list of non terminals in the grammar"""
    if is_annotated(t):
        return all([is_terminal(inner, l) for inner in get_generic_parameters(t)])
    if not get_arguments(t):
        return True
    return t not in l
