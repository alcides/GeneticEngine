from typing import Protocol, List
from geneticengine.core.utils import get_arguments


class TreeNode(Protocol):
    depth: int
    distance_to_term: int
    nodes: int


class PrettyPrintable(object):
    def __repr__(self):
        args = ", ".join(
            [f"{a}={getattr(self, a)}" for (a, at) in get_arguments(self.__class__)]
        )
        return f"{self.__class__.__name__}({args})"