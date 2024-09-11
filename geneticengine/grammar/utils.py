from __future__ import annotations

import inspect
from abc import ABC
from typing import Any, get_origin, Union
from typing import get_type_hints
from typing import Protocol
from typing import TYPE_CHECKING

from geneticengine.grammar.decorators import get_gengy

if TYPE_CHECKING:
    from geneticengine.solutions.tree import GengyList


def has_annotated_mutation(ty: type[Any]):
    """Returns whether type has an annotated mutation within metadata."""
    if hasattr(ty, "__metadata__"):
        if hasattr(ty.__metadata__[0], "mutate"):
            return True
    return False


def has_annotated_crossover(ty: type[Any]):
    """Returns whether type has an annotated crossover within metadata."""
    if hasattr(ty, "__metadata__"):
        if hasattr(ty.__metadata__[0], "crossover"):
            return True
    return False


def is_annotated(ty: type[Any]):
    """Returns whether type is annotated with metadata."""
    return hasattr(ty, "__metadata__")


def is_generic_list(ty: type[Any]):
    """Returns whether a type is List[T] for any T."""
    return hasattr(ty, "__origin__") and ty.__origin__ is list


def is_generic_tuple(ty: type[Any]):
    """Returns whether a type is tuple[X, Y, ...] for any X, Y, ...."""
    return hasattr(ty, "__origin__") and ty.__origin__ is tuple


def is_generic(ty: type[Any]):
    """Returns whether a type is x[T] for any T."""
    return hasattr(ty, "__origin__")


def is_union(ty: type[Any]):
    """Returns whether a type is List[T] for any T."""
    return get_origin(ty) is Union


def get_generic_parameters(ty: type[Any]) -> list[type]:
    """Annotated[T, <annotations>] or List[T], this function returns
    Dict[T,]"""
    return ty.__args__


def get_generic_parameter(ty: type[Any]) -> type:
    """When given Annotated[T, <annotations>] or List[T], this function returns
    T."""
    return get_generic_parameters(ty)[0]


def strip_annotations(ty: type[Any]) -> type:
    """When given Annotated[T, <annotations>] or List[T], this function
    recurses with T Otherwise, it returns the parameter unchanged."""
    if is_generic_list(ty) or is_annotated(ty):
        return strip_annotations(get_generic_parameter(ty))
    else:
        return ty


def has_arguments(n: Any) -> bool:
    """Returns whether a node has arguments or not."""
    return hasattr(n, "__init__") and hasattr(n.__init__, "__annotations__") and len(n.__init__.__annotations__) > 0


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
    elif isinstance(n, GengyList):
        return [(f"{i}", n.typ) for i in range(len(n))]
    return []


def is_abstract(t: type) -> bool:
    """Returns whether a class is a Protocol or AbstractBaseClass."""
    if is_union(t):
        return False
    return t.mro()[1] in [ABC, Protocol] or get_gengy(t).get("abstract", False)


def is_terminal(t: type, non_terminals: set[type]) -> bool:
    """Returns whether a node is a terminal or not, based on the list of non
    terminals in the grammar."""
    if is_annotated(t):
        return all(
            [is_terminal(inner, non_terminals) for inner in get_generic_parameters(t)],
        )
    return t not in non_terminals


def all_init_arguments_typed(t: type) -> bool:
    if hasattr(t, "__init__"):
        m = getattr(t, "__init__")
        d = inspect.getfullargspec(m)
        return all(x in d.annotations for x in d.args[1:])  # starts with self
    return False


def strip_dependencies(s: str) -> str:
    return s.split(".")[-1].split("'")[0]


def is_builtin_class_instance(obj):
    return obj.__class__.__module__ == "builtins"


def is_metahandler(ty: type) -> bool:
    """Returns if type is a metahandler. AnnotatedType[int, IntRange(3,10)] is
    an example of a Metahandler.

    Verification is done using the __metadata__, which is the first
    argument of Annotated
    """
    return hasattr(ty, "__metadata__")
