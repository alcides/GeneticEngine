from abc import ABC
from typing import (
    Any,
    Protocol,
    Type,
    Tuple,
    List,
)


def is_annotated(ty: Type[Any]):
    """ Returns whether type is annotated with metadata. """
    return hasattr(ty, "__metadata__")


def is_generic_list(ty: Type[Any]):
    """ Returns whether a type is List[T] for any T """
    return hasattr(ty, "__origin__") and ty.__origin__ is list


def get_generic_parameters(ty: Type[Any]) -> list[type]:
    """ When given Dict[T, R], this function returns [T, R]"""
    return ty.__args__


def get_generic_parameter(ty: Type[Any]) -> type:
    """ When given List[T], this function returns T"""
    return get_generic_parameters(ty)[0]


def get_arguments(n) -> List[Tuple[str, type]]:
    if hasattr(n, "__annotations__"):
        args = n.__annotations__
        return [(a, args[a]) for a in args]
    else:
        return []


def is_abstract(t: type) -> bool:
    """ Returns whether a class is a Protocol or AbstractBaseClass """
    return t.mro()[1] in [ABC, Protocol]  # TODO: Protocol not working


def is_terminal(t: type, l: list[type]) -> bool:
    """ Returns whether a node is a terminal or not, based on the list of non terminals in the grammar """
    if is_annotated(t):
        return all([is_terminal(inner, l) for inner in get_generic_parameters(t)])
    if not get_arguments(t):
        return True
    return t not in l