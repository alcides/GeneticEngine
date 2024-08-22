from __future__ import annotations

from typing import Any, Generic, TypeVar
from typing import Protocol
from typing import runtime_checkable

from geneticengine.grammar.utils import get_arguments
from geneticengine.representations.tree.initializations import LocalSynthesisContext


@runtime_checkable
class TreeNode(Protocol):
    gengy_labeled: bool
    gengy_distance_to_term: int
    gengy_nodes: int
    gengy_weighted_nodes: int
    gengy_types_this_way: dict[type, list[Any]]
    gengy_init_values: tuple[Any]
    gengy_synthesis_context: dict[str, "TreeNode"]


class PrettyPrintable:
    def __repr__(self):
        args = ", ".join(
            [f"{a}={getattr(self, a)}" for (a, at) in get_arguments(self.__class__)],
        )
        return f"{self.__class__.__name__}({args})"


T = TypeVar("T")


class GengyList(list, Generic[T]):

    gengy_synthesis_context : LocalSynthesisContext

    def __init__(self, typ, vals):
        super().__init__(vals)
        self.typ = typ
        self.gengy_init_values = vals

    def new_like(self, *newargs) -> GengyList[T]:
        n: GengyList[T] = GengyList(self.typ, newargs)
        if hasattr(self, "gengy_synthesis_context"):
            n.gengy_synthesis_context = self.gengy_synthesis_context
        return n

    def __hash__(self): # pyright: ignore
        return sum(hash(o) for o in self)

    def __add__(self, value) -> GengyList[T]:
        v = super().__add__(value)
        return GengyList(self.typ, v)
