from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Protocol
from typing import runtime_checkable

from geneticengine.core.utils import get_arguments


@runtime_checkable
class TreeNode(Protocol):
    gengy_labeled: bool
    gengy_distance_to_term: int
    gengy_nodes: int
    gengy_types_this_way: dict[type, list[Any]]
    gengy_init_values: tuple[Any]


class PrettyPrintable:
    def __repr__(self):
        args = ", ".join(
            [f"{a}={getattr(self, a)}" for (a, at) in get_arguments(self.__class__)],
        )
        return f"{self.__class__.__name__}({args})"
