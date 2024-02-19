from __future__ import annotations

from typing import Any, Generic, TypeVar
from typing import Protocol
from typing import runtime_checkable

from geneticengine.grammar.utils import get_arguments


@runtime_checkable
class TreeNode(Protocol):
    gengy_labeled: bool
    gengy_distance_to_term: int
    gengy_nodes: int
    gengy_weighted_nodes: int
    gengy_types_this_way: dict[type, list[Any]]
    gengy_init_values: tuple[Any]


class PrettyPrintable:
    def __repr__(self):
        args = ", ".join(
            [f"{a}={getattr(self, a)}" for (a, at) in get_arguments(self.__class__)],
        )
        return f"{self.__class__.__name__}({args})"


T = TypeVar("T")


class GengyList(list, Generic[T]):
    def __init__(self, typ, vals):
        super().__init__(vals)
        self.typ = typ
        self.gengy_init_values = vals

    def new_like(self, *newargs):
        return GengyList(self.typ, newargs)

    def __hash__(self):
        return sum(hash(o) for o in self)
