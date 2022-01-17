from dataclasses import dataclass
from typing import Callable, Type, Any

WrapperType = Callable[[int, Type], Any]


@dataclass
class Wrapper(object):
    depth: int
    target: Type
