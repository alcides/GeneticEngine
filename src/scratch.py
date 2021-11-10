from typing import Annotated
from geneticengine.core.utils import (
    get_arguments,
    is_generic_list,
    get_generic_parameters,
)


class B:
    x: Annotated[int, True]


class A:
    y: Annotated[B, True]


def is_terminal(t: type, l: list[type]) -> bool:
    if hasattr(t, "__metadata__"):
        return all([is_terminal(inner, l) for inner in get_generic_parameters(t)])
    return t not in l


l = [A, B]

print(is_terminal(A, l))
print(is_terminal(B, l))
print(is_terminal(Annotated[B, True], l))
print("_")
print(is_terminal(int, l))
print(is_terminal(Annotated[int, True], l))
