from typing import Protocol, runtime_checkable, List, Dict, Any

from geneticengine.core.utils import get_arguments


@runtime_checkable
class TreeNode(Protocol):
    gengy_labeled: bool
    gengy_distance_to_term: int
    gengy_nodes: int
    gengy_types_this_way: Dict[type, List[Any]]


class PrettyPrintable(object):
    def __repr__(self):
        args = ", ".join(
            [f"{a}={getattr(self, a)}" for (a, at) in get_arguments(self.__class__)]
        )
        return f"{self.__class__.__name__}({args})"
