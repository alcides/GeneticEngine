from typing import Generic, TypeVar


C = TypeVar("C")


class GenericWrapper(Generic[C]):
    container: dict[str, C]

    def __init__(self, c: C):
        self.container = {"": c}

    def get(self) -> C:
        return self.container[""]
